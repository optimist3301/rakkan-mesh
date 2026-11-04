package mesh

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base32"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mldht"
)

// Bootstrap through the BitTorrent Mainline DHT (package mldht), which
// nobody runs and nobody can take down. A node that knows only the
// network's ID finds members there:
//
//   - the root's record (signed by the root key, which the network ID
//     is) lists announcers: members, by peer ID, chosen by the root;
//   - each announcer's record (signed by its own node key, which its peer
//     ID is) gives its current addresses.
//
// The root signs its record rarely (rakkan-node mainline-publish); any
// member stores it again, unchanged, so it stays while the network lives.
// Announcers republish their addresses. Both records are sealed with a key
// derived from the network ID, and stored under salts derived from it: a
// DHT node that does not know the network learns neither who the members
// are nor where. Signatures are checked as always, and the connection
// then authenticates the member as any connection does.

// mainlineRecords are a network's salts and sealing key.
type mainlineRecords struct {
	rootSalt, addrSalt []byte
	aead               cipher.AEAD
}

func newMainlineRecords(netID string) *mainlineRecords {
	h := sha256.Sum256([]byte("rakkan-mesh/mainline/v1\x00" + netID))
	tag := base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(h[:15])
	key := sha256.Sum256(append([]byte("seal\x00"), h[:]...))
	block, _ := aes.NewCipher(key[:])
	aead, _ := cipher.NewGCM(block)
	return &mainlineRecords{rootSalt: []byte("rkn/b/" + tag), addrSalt: []byte("rkn/a/" + tag), aead: aead}
}

// seal encrypts a record's content, bound to its salt.
func (r *mainlineRecords) seal(salt, plain []byte) []byte {
	nonce := make([]byte, r.aead.NonceSize())
	rand.Read(nonce)
	return r.aead.Seal(nonce, nonce, plain, salt)
}

func (r *mainlineRecords) open(salt, sealed []byte) ([]byte, error) {
	n := r.aead.NonceSize()
	if len(sealed) < n {
		return nil, errors.New("mesh: short mainline record")
	}
	return r.aead.Open(nil, sealed[:n], sealed[n:], salt)
}

// maxAnnouncers fits a root record in an item.
const maxAnnouncers = 28

func edKey(k crypto.PubKey) (ed25519.PublicKey, error) {
	if k.Type() != crypto.Ed25519 {
		return nil, errors.New("mesh: not an Ed25519 key")
	}
	raw, err := k.Raw()
	return ed25519.PublicKey(raw), err
}

func edPrivate(k crypto.PrivKey) (ed25519.PrivateKey, error) {
	if k.Type() != crypto.Ed25519 {
		return nil, errors.New("mesh: not an Ed25519 key")
	}
	raw, err := k.Raw()
	return ed25519.PrivateKey(raw), err
}

// SignBootstrapRecord is the root's record naming the announcers, with a
// sequence number above any earlier one's (the time, say).
func SignBootstrapRecord(root crypto.PrivKey, n *membership.Network, announcers []peer.ID, seq int64) (*mldht.Item, error) {
	if len(announcers) == 0 || len(announcers) > maxAnnouncers {
		return nil, fmt.Errorf("mesh: %d announcers, from 1 to %d", len(announcers), maxAnnouncers)
	}
	plain := []byte{1, byte(len(announcers))}
	for _, id := range announcers {
		pub, err := id.ExtractPublicKey()
		if err != nil {
			return nil, fmt.Errorf("mesh: announcer %s: %w", id, err)
		}
		raw, err := edKey(pub)
		if err != nil {
			return nil, fmt.Errorf("mesh: announcer %s: %w", id, err)
		}
		plain = append(plain, raw...)
	}
	sk, err := edPrivate(root)
	if err != nil {
		return nil, err
	}
	rec := newMainlineRecords(n.ID)
	return mldht.Sign(sk, rec.rootSalt, seq, rec.seal(rec.rootSalt, plain))
}

func parseAnnouncers(plain []byte) ([]peer.ID, error) {
	if len(plain) < 2 || plain[0] != 1 || len(plain) != 2+32*int(plain[1]) {
		return nil, errors.New("mesh: malformed root record")
	}
	var ids []peer.ID
	for i := range int(plain[1]) {
		pub, err := crypto.UnmarshalEd25519PublicKey(plain[2+32*i : 2+32*(i+1)])
		if err != nil {
			return nil, err
		}
		id, err := peer.IDFromPublicKey(pub)
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func encodeAddrs(addrs []ma.Multiaddr) []byte {
	b := []byte{1}
	for _, a := range addrs {
		raw := a.Bytes()
		if len(b)+binary.MaxVarintLen16+len(raw) > mldht.MaxValue-28-16 {
			break // what fits, sealed, in an item
		}
		b = binary.AppendUvarint(b, uint64(len(raw)))
		b = append(b, raw...)
	}
	return b
}

func decodeAddrs(b []byte) ([]ma.Multiaddr, error) {
	if len(b) == 0 || b[0] != 1 {
		return nil, errors.New("mesh: malformed address record")
	}
	b = b[1:]
	var out []ma.Multiaddr
	for len(b) > 0 {
		n, k := binary.Uvarint(b)
		if k <= 0 || n > uint64(len(b)-k) {
			return nil, errors.New("mesh: malformed address record")
		}
		a, err := ma.NewMultiaddrBytes(b[k : k+int(n)])
		if err == nil {
			out = append(out, a)
		}
		b = b[k+int(n):]
	}
	return out, nil
}

// mainlineLoop keeps the network's records in the DHT and, while the node
// has few members, finds more there.
func (n *Node) mainlineLoop() {
	boot := n.cfg.MainlineBootstrap
	if len(boot) == 0 {
		boot = mldht.DefaultBootstrap
	}
	c, err := mldht.Listen(":0", boot)
	if err != nil {
		n.log.Warn("mainline DHT", "err", err)
		return
	}
	go func() {
		<-n.ctx.Done()
		c.Close()
	}()
	rec := newMainlineRecords(n.cfg.Network.ID)
	rootKey, err := edKey(n.cfg.Network.RootKey())
	if err != nil {
		n.log.Warn("mainline DHT", "err", err)
		return
	}
	var root *mldht.Item // the newest root record seen
	for {
		connected := 0
		for _, m := range n.Members() {
			if m.Connected {
				connected++
			}
		}
		ctx, cancel := context.WithTimeout(n.ctx, 2*time.Minute)
		root = n.mainlineRound(ctx, c, rec, rootKey, root, connected < 3)
		cancel()
		wait := n.cfg.MainlineInterval
		if connected == 0 {
			wait = min(wait, time.Minute) // alone: look again soon
		}
		select {
		case <-n.ctx.Done():
			return
		case <-time.After(wait):
		}
	}
}

// mainlineRound reads the root record and stores it again, publishes the
// node's addresses if it is an announcer, and if find, dials the
// announcers.
func (n *Node) mainlineRound(ctx context.Context, c *mldht.Client, rec *mainlineRecords, rootKey ed25519.PublicKey, root *mldht.Item, find bool) *mldht.Item {
	var minSeq int64
	if root != nil {
		minSeq = root.Seq
	}
	if it, err := c.Get(ctx, rootKey, rec.rootSalt, minSeq); err == nil {
		root = it
	} else if root == nil {
		n.log.Debug("mainline DHT: no root record", "err", err)
		return nil
	}
	plain, err := rec.open(rec.rootSalt, root.Value)
	if err != nil {
		n.log.Warn("mainline DHT: root record", "err", err)
		return nil
	}
	announcers, err := parseAnnouncers(plain)
	if err != nil {
		n.log.Warn("mainline DHT: root record", "err", err)
		return nil
	}
	var wg sync.WaitGroup
	if n.Certified() {
		// Members keep the root's record alive.
		wg.Go(func() { c.Put(ctx, root) })
	}
	self := n.host.ID()
	if slices.Contains(announcers, self) {
		wg.Go(func() {
			sk, err := edPrivate(n.cfg.Key)
			if err != nil {
				n.log.Warn("mainline DHT: cannot announce", "err", err)
				return
			}
			addrs := n.advertisedAddrs(n.host.Addrs())
			it, err := mldht.Sign(sk, rec.addrSalt, time.Now().Unix(), rec.seal(rec.addrSalt, encodeAddrs(addrs)))
			if err == nil {
				_, err = c.Put(ctx, it)
			}
			if err != nil {
				n.log.Warn("mainline DHT: announcing", "err", err)
			}
		})
	}
	if find {
		wg.Go(func() {
			for _, ai := range findAnnouncers(ctx, c, rec, announcers, func(id peer.ID) bool { return id == self || n.isMember(id) }) {
				n.log.Info("mainline DHT: found an announcer", "peer", ai.ID.ShortString(), "addrs", len(ai.Addrs))
				n.addDialTarget(ai)
			}
		})
	}
	wg.Wait()
	return root
}

// findAnnouncers reads the address records of the announcers but those
// skipped, at once.
func findAnnouncers(ctx context.Context, c *mldht.Client, rec *mainlineRecords, announcers []peer.ID, skip func(peer.ID) bool) []peer.AddrInfo {
	var mu sync.Mutex
	var wg sync.WaitGroup
	var out []peer.AddrInfo
	for _, id := range announcers {
		if skip != nil && skip(id) {
			continue
		}
		wg.Go(func() {
			pub, err := id.ExtractPublicKey()
			if err != nil {
				return
			}
			key, err := edKey(pub)
			if err != nil {
				return
			}
			it, err := c.Get(ctx, key, rec.addrSalt, 0)
			if err != nil {
				return
			}
			plain, err := rec.open(rec.addrSalt, it.Value)
			if err != nil {
				return
			}
			addrs, err := decodeAddrs(plain)
			if err == nil && len(addrs) > 0 {
				mu.Lock()
				out = append(out, peer.AddrInfo{ID: id, Addrs: addrs})
				mu.Unlock()
			}
		})
	}
	wg.Wait()
	return out
}

// FindAnnouncers looks up a network's announcers and their addresses in
// the Mainline DHT, from its entry points (mldht.DefaultBootstrap if
// none): what a node needs when it knows nothing but the network, a guest
// asking for the policy, say. The addresses are the announcers' own
// signed word; whoever answers there still has to prove membership.
func FindAnnouncers(ctx context.Context, n *membership.Network, dhtBootstrap []string) ([]peer.AddrInfo, error) {
	if len(dhtBootstrap) == 0 {
		dhtBootstrap = mldht.DefaultBootstrap
	}
	c, err := mldht.Listen(":0", dhtBootstrap)
	if err != nil {
		return nil, err
	}
	defer c.Close()
	rootKey, err := edKey(n.RootKey())
	if err != nil {
		return nil, err
	}
	rec := newMainlineRecords(n.ID)
	root, err := c.Get(ctx, rootKey, rec.rootSalt, 0)
	if errors.Is(err, mldht.ErrNotFound) {
		return nil, errors.New("mesh: the network is not in the Mainline DHT: its admin has not run rakkan-node network announce, or nobody has kept the record there")
	}
	if err != nil {
		return nil, fmt.Errorf("mesh: the network's record in the Mainline DHT: %w", err)
	}
	plain, err := rec.open(rec.rootSalt, root.Value)
	if err != nil {
		return nil, err
	}
	announcers, err := parseAnnouncers(plain)
	if err != nil {
		return nil, err
	}
	found := findAnnouncers(ctx, c, rec, announcers, nil)
	if len(found) == 0 {
		return nil, fmt.Errorf("mesh: none of the %d announcers has an address record", len(announcers))
	}
	return found, nil
}
