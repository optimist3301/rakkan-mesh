// Package ygg embeds a Yggdrasil node (yggdrasil-go's core, no TUN device,
// no root rights) and carries libp2p connections over it.
//
// Yggdrasil routes by public key across an overlay of peerings. A node
// behind NAT makes outbound peerings to a few public Yggdrasil nodes and is
// then reachable by its key from any other node of the overlay, so two
// rakkan nodes that both sit behind NAT can still connect, with no relay
// of ours and no server with a public IP.
//
// Traffic between two nodes is QUIC over Yggdrasil's datagrams (see
// transport.go); libp2p then secures and multiplexes it as it does TCP.
package ygg

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha512"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"errors"
	"fmt"
	"log/slog"
	"math/big"
	mrand "math/rand/v2"
	"net"
	"net/url"
	"sync"
	"time"

	iwt "github.com/Arceliar/ironwood/types"
	"github.com/libp2p/go-libp2p/core/crypto"
	ma "github.com/multiformats/go-multiaddr"
	"github.com/quic-go/quic-go"
	"github.com/yggdrasil-network/yggdrasil-go/src/core"
)

// DefaultPeers are public Yggdrasil nodes with good uptime on
// publicpeers.neilalexander.dev (2026-11), by several operators in several
// countries. A node peers with DefaultPeerCount of them, picked at random,
// so that no single one carries everybody.
var DefaultPeers = []string{
	"tls://yggdrasil.neilalexander.dev:64648",
	"tls://ygg1.mk16.de:1338",
	"tls://ygg2.mk16.de:1338",
	"tls://yggdrasil.su:62586",
	"tls://de.wwire.su:2231",
	"tls://wwire.su:2231",
	"tls://kursk.cleverfox.org:15015",
	"tls://yg-ovb.magicum.net:32333",
	"tls://ygg.mkg20001.io:443",
	"tls://nl1.route172.de:443",
}

// DefaultPeerCount is how many of DefaultPeers a node peers with.
const DefaultPeerCount = 3

// Config sets up the embedded node.
type Config struct {
	// Peers are Yggdrasil peer URIs (tls://host:port, tcp://, quic://, …)
	// to connect to. Nil means DefaultPeerCount random DefaultPeers; an
	// empty non-nil slice means none, for a node that only listens.
	Peers []string
	// Listen are URIs to accept peerings on, e.g. tls://0.0.0.0:0. A node
	// behind NAT needs none.
	Listen []string
	Logger *slog.Logger
}

// Node is a running embedded Yggdrasil node.
type Node struct {
	core *core.Core
	key  ed25519.PublicKey
	tr   *quic.Transport
	tls  *tls.Config
	log  *slog.Logger

	mu       sync.Mutex
	listener *quic.Listener
	closed   bool
	ctx      context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	// paths and pongs wake dials waiting for a path to a key to be found
	// and for a key to answer a ping (session.go).
	paths, pongs signals
}

// DeriveKey derives the node's Yggdrasil key from its libp2p key. The two
// are separate keys, so neither protocol ever signs with the other's key,
// but the Yggdrasil address stays the same across restarts.
func DeriveKey(key crypto.PrivKey) (ed25519.PrivateKey, error) {
	raw, err := key.Raw()
	if err != nil {
		return nil, err
	}
	h := sha512.New()
	h.Write([]byte("rakkan-mesh yggdrasil key v1\x00"))
	h.Write(raw)
	return ed25519.NewKeyFromSeed(h.Sum(nil)[:ed25519.SeedSize]), nil
}

// Start starts an embedded Yggdrasil node with a key derived from key.
func Start(key crypto.PrivKey, cfg Config) (*Node, error) {
	sk, err := DeriveKey(key)
	if err != nil {
		return nil, fmt.Errorf("ygg: %w", err)
	}
	cert, err := selfSigned(sk)
	if err != nil {
		return nil, fmt.Errorf("ygg: %w", err)
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	peers := cfg.Peers
	if peers == nil {
		peers = pick(DefaultPeers, DefaultPeerCount)
	}
	var opts []core.SetupOption
	for _, p := range peers {
		if _, err := url.Parse(p); err != nil {
			return nil, fmt.Errorf("ygg: peer %q: %w", p, err)
		}
		opts = append(opts, core.Peer{URI: p})
	}
	for _, l := range cfg.Listen {
		opts = append(opts, core.ListenAddress(l))
	}
	c, err := core.New(cert, logger{log.With("component", "yggdrasil")}, opts...)
	if err != nil {
		return nil, fmt.Errorf("ygg: %w", err)
	}
	n := &Node{
		core: c,
		key:  sk.Public().(ed25519.PublicKey),
		log:  log,
		tls:  quicTLS(cert),
	}
	n.ctx, n.cancel = context.WithCancel(context.Background())
	c.SetPathNotify(n.paths.notify)
	n.tr = &quic.Transport{Conn: c}
	n.wg.Add(2)
	go n.ctlLoop()
	go n.nudgeLoop()
	log.Info("yggdrasil started", "address", c.Address(), "key", fmt.Sprintf("%x", n.key), "peers", len(peers))
	return n, nil
}

// PublicKey is the node's Yggdrasil key.
func (n *Node) PublicKey() ed25519.PublicKey { return n.key }

// Address is the node's Yggdrasil IPv6 address (200::/7), for logs.
func (n *Node) Address() net.IP { return n.core.Address() }

// Multiaddr is the address other nodes dial this one at: /ygg/<key>.
func (n *Node) Multiaddr() ma.Multiaddr { return Multiaddr(n.key) }

// PeersUp counts the node's peerings that are up.
func (n *Node) PeersUp() int {
	up := 0
	for _, p := range n.core.GetPeers() {
		if p.Up {
			up++
		}
	}
	return up
}

// AddPeer connects to another Yggdrasil node, e.g. tls://host:port.
func (n *Node) AddPeer(uri string) error {
	u, err := url.Parse(uri)
	if err != nil {
		return err
	}
	return n.core.AddPeer(u, "")
}

// Listen starts accepting peerings on uri (tcp://127.0.0.1:0, …) and
// returns the address it listens on.
func (n *Node) Listen(uri string) (net.Addr, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	l, err := n.core.Listen(u, "")
	if err != nil {
		return nil, err
	}
	return l.Addr(), nil
}

// Close stops the node and every connection over it.
func (n *Node) Close() error {
	n.mu.Lock()
	if n.closed {
		n.mu.Unlock()
		return nil
	}
	n.closed = true
	n.mu.Unlock()
	n.cancel()
	n.wg.Wait()
	err := n.tr.Close()
	n.core.Stop()
	return err
}

// dial opens a QUIC connection to the node with Yggdrasil key to.
func (n *Node) dial(ctx context.Context, to ed25519.PublicKey) (*quic.Conn, error) {
	if to.Equal(n.key) {
		return nil, errors.New("ygg: dialing self")
	}
	if err := n.waitPath(ctx, to); err != nil {
		return nil, err
	}
	if err := n.waitSession(ctx, to); err != nil {
		return nil, err
	}
	return n.tr.Dial(ctx, iwt.Addr(to), n.tls, quicConfig())
}

// listen returns the node's QUIC listener, starting it on first use. A
// quic.Transport serves one listener, shared by every libp2p listener.
func (n *Node) listen() (*quic.Listener, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return nil, net.ErrClosed
	}
	if n.listener == nil {
		l, err := n.tr.Listen(n.tls, quicConfig())
		if err != nil {
			return nil, err
		}
		n.listener = l
	}
	return n.listener, nil
}

// unlisten forgets the listener l after closing it, so a later listen
// starts a new one.
func (n *Node) unlisten(l *quic.Listener) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.listener == l {
		n.listener = nil
	}
	l.Close()
}

func quicConfig() *quic.Config {
	return &quic.Config{
		// Paths over Yggdrasil may take a few seconds to be found.
		HandshakeIdleTimeout: 20 * time.Second,
		MaxIdleTimeout:       time.Minute,
		KeepAlivePeriod:      15 * time.Second,
		// Each connection carries one stream, which libp2p multiplexes.
		MaxIncomingStreams:    1,
		MaxIncomingUniStreams: -1,
	}
}

// alpn names the protocol inside QUIC: one stream that libp2p upgrades.
const alpn = "rakkan-ygg/1"

// quicTLS is the TLS config of the QUIC layer. It checks no certificate:
// Yggdrasil already authenticates the remote key, and libp2p's security
// handshake on top authenticates the peer ID.
func quicTLS(cert *tls.Certificate) *tls.Config {
	return &tls.Config{
		Certificates:       []tls.Certificate{*cert},
		InsecureSkipVerify: true,
		ClientAuth:         tls.NoClientCert,
		NextProtos:         []string{alpn},
		MinVersion:         tls.VersionTLS13,
	}
}

func selfSigned(sk ed25519.PrivateKey) (*tls.Certificate, error) {
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 64))
	if err != nil {
		return nil, err
	}
	tmpl := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: fmt.Sprintf("%x", sk.Public())},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(10 * 365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, sk.Public(), sk)
	if err != nil {
		return nil, err
	}
	return &tls.Certificate{Certificate: [][]byte{der}, PrivateKey: sk}, nil
}

func pick(all []string, k int) []string {
	p := append([]string(nil), all...)
	mrand.Shuffle(len(p), func(i, j int) { p[i], p[j] = p[j], p[i] })
	return p[:min(k, len(p))]
}

// logger adapts slog to Yggdrasil's logger. Its info messages go to debug:
// they describe every peering and are noise in the node's log.
type logger struct{ l *slog.Logger }

func (g logger) Printf(f string, a ...any) { g.l.Debug(fmt.Sprintf(f, a...)) }
func (g logger) Println(a ...any)          { g.l.Debug(fmt.Sprint(a...)) }
func (g logger) Infof(f string, a ...any)  { g.l.Debug(fmt.Sprintf(f, a...)) }
func (g logger) Infoln(a ...any)           { g.l.Debug(fmt.Sprint(a...)) }
func (g logger) Warnf(f string, a ...any)  { g.l.Warn(fmt.Sprintf(f, a...)) }
func (g logger) Warnln(a ...any)           { g.l.Warn(fmt.Sprint(a...)) }
func (g logger) Errorf(f string, a ...any) { g.l.Error(fmt.Sprintf(f, a...)) }
func (g logger) Errorln(a ...any)          { g.l.Error(fmt.Sprint(a...)) }
func (g logger) Debugf(f string, a ...any) { g.l.Debug(fmt.Sprintf(f, a...)) }
func (g logger) Debugln(a ...any)          { g.l.Debug(fmt.Sprint(a...)) }
func (g logger) Traceln(a ...any)          {}
