package mesh

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"sync"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/sealed"
)

// Sealed reports: members hand the network data that only the holder of the
// root key may read. An application gives its node a report; the node
// seals it to the network's root key (the network ID embeds it), signs the
// result with its certificate, gossips it, and every certified member keeps
// it in a bounded spool and offers it to others. Relays and spools see the
// author and the size, never the content. The root, whenever it comes
// online, takes the packets from any node and opens them with its key: the
// node never has that key.

const (
	// MaxReport is the largest report text a node accepts.
	MaxReport = 192 << 10
	// maxPacket bounds a packet on the wire: report, seal and certificate.
	maxPacket = MaxReport + 16<<10

	reportDomain  = "rakkan-report-v1\x00"
	reportsSync   = protocol.ID("/rakkan/reports/1.0.0")
	reportMaxAge  = 30 * 24 * time.Hour
	reportSkew    = 10 * time.Minute
	reportRate    = 60 // per author per minute, from the network
	guestRate     = 4  // the same for a guest, who costs the network no certificate
	pullMaxCount  = 256
	pullMaxBytes  = 8 << 20
	pullHaveLimit = 4096
)

// ReportsConfig enables sealed reports; see Config.Reports.
type ReportsConfig struct {
	// Dir is the spool directory. Empty disables keeping and sending
	// reports; the node still relays them.
	Dir string
	// MaxBytes bounds the spool (default 64 MiB), MaxCount its packets
	// (default 4096, what one sync round can describe).
	MaxBytes int64
	MaxCount int
	// SyncInterval is how often the node pulls what it lacks from a random
	// member (default 30s).
	SyncInterval time.Duration
}

// ReportPacket is a report as it travels and is stored: sealed, and signed
// by its author: a member by its certificate, a guest by the key its proof
// of work is made for.
type ReportPacket struct {
	Version int                    `json:"v"`
	Cert    *membership.SignedCert `json:"cert,omitempty"`
	Guest   *GuestAuthor           `json:"guest,omitempty"` // instead of Cert
	Time    int64                  `json:"time"`            // Unix ms
	Sig     []byte                 `json:"sig"`             // by the author's key, over Time and Sealed
	Sealed  []byte                 `json:"sealed"`
}

// GuestAuthor identifies a guest: its public key (libp2p, protobuf) and the
// proof of work that admits it. Anyone can check both, the proof against the
// work the network's policy asks.
type GuestAuthor struct {
	Key   []byte           `json:"key"`
	Proof membership.Proof `json:"proof"`
}

func packetDigest(t int64, sealedBytes []byte) []byte {
	b := binary.BigEndian.AppendUint64([]byte(reportDomain), uint64(t))
	return append(b, sealedBytes...)
}

// VerifyReport checks a packet's structure, its certificate and its
// signature, and returns the author's certificate. revoked may be nil.
// Guests' packets are refused; see VerifyReportGuests.
func VerifyReport(nw *membership.Network, data []byte, now time.Time, revoked func(string) bool) (*ReportPacket, *membership.Cert, error) {
	return VerifyReportGuests(nw, data, now, revoked, -1)
}

// VerifyReportGuests is VerifyReport that also takes a guest's packet when
// the author's proof does at least guestBits of work (0 takes any proof;
// below 0 refuses guests). For a guest the returned certificate is
// synthesized, as the node does for a connected guest: it carries
// membership.RoleGuest and nothing else.
func VerifyReportGuests(nw *membership.Network, data []byte, now time.Time, revoked func(string) bool, guestBits int) (*ReportPacket, *membership.Cert, error) {
	if len(data) > maxPacket {
		return nil, nil, errors.New("report too large")
	}
	var p ReportPacket
	if err := json.Unmarshal(data, &p); err != nil || p.Version != 1 || len(p.Sealed) < sealed.Overhead || len(p.Sealed) > MaxReport+sealed.Overhead {
		return nil, nil, errors.New("malformed report")
	}
	var (
		c  *membership.Cert
		pk crypto.PubKey
	)
	switch {
	case p.Cert != nil && p.Guest == nil:
		var err error
		if c, err = nw.Verify(p.Cert, now, revoked); err != nil {
			return nil, nil, err
		}
		if pk, err = c.Subject.ExtractPublicKey(); err != nil {
			return nil, nil, err
		}
	case p.Cert == nil && p.Guest != nil:
		if guestBits < 0 {
			return nil, nil, errors.New("this network takes no reports from guests")
		}
		var err error
		if pk, err = crypto.UnmarshalPublicKey(p.Guest.Key); err != nil {
			return nil, nil, errors.New("malformed report")
		}
		id, err := peer.IDFromPublicKey(pk)
		if err != nil {
			return nil, nil, err
		}
		if revoked != nil && revoked(id.String()) {
			return nil, nil, membership.ErrRevoked
		}
		if w := nw.Work(id, p.Guest.Proof); w < guestBits {
			return nil, nil, fmt.Errorf("proof of work of %d bits, the network wants %d", w, guestBits)
		}
		c = &membership.Cert{Version: 1, Network: nw.ID, Subject: id, Roles: []string{membership.RoleGuest}}
	default:
		return nil, nil, errors.New("malformed report")
	}
	if ok, err := pk.Verify(packetDigest(p.Time, p.Sealed), p.Sig); err != nil || !ok {
		return nil, nil, errors.New("bad report signature")
	}
	if t := time.UnixMilli(p.Time); t.After(now.Add(reportSkew)) || t.Before(now.Add(-reportMaxAge)) {
		return nil, nil, errors.New("report time out of range")
	}
	return &p, c, nil
}

// ReportContext is the context a report is sealed under for a network.
func ReportContext(networkID string) []byte { return []byte("rakkan-report " + networkID) }

func rootEd25519(nw *membership.Network) (ed25519.PublicKey, error) {
	raw, err := nw.RootKey().Raw()
	if err != nil {
		return nil, err
	}
	return ed25519.PublicKey(raw), nil
}

// verifyReport checks a packet against the network and the policy in force:
// a guest's report counts while the network is open and its proof is enough.
func (n *Node) verifyReport(data []byte) (*ReportPacket, *membership.Cert, error) {
	bits := -1
	if _, pol := n.policies.Current(); pol.Open {
		bits = pol.PowBits
	}
	return VerifyReportGuests(n.cfg.Network, data, time.Now(), n.revs.Revoked, bits)
}

func (n *Node) reportsTopic() string { return "/rakkan/" + n.tag() + "/reports/1" }

func (n *Node) startReports() error {
	rc := n.cfg.Reports
	if rc.MaxBytes <= 0 {
		rc.MaxBytes = 64 << 20
	}
	if rc.MaxCount <= 0 {
		rc.MaxCount = pullHaveLimit
	}
	if rc.SyncInterval <= 0 {
		rc.SyncInterval = 30 * time.Second
	}
	if rc.Dir != "" {
		sp, err := openSpool(rc.Dir, rc.MaxBytes, rc.MaxCount, reportMaxAge)
		if err != nil {
			return fmt.Errorf("mesh: reports: %w", err)
		}
		n.spool = sp
	}
	rate := newRateLimit(reportRate, time.Minute)
	guestLimit := newRateLimit(guestRate, time.Minute)
	topic := n.reportsTopic()
	err := n.ps.RegisterTopicValidator(topic, func(_ context.Context, _ peer.ID, m *pubsub.Message) pubsub.ValidationResult {
		_, c, err := n.verifyReport(m.Data)
		if err != nil {
			return pubsub.ValidationReject
		}
		// The message is signed by whoever published it, and a relay may
		// forward a packet of another author; only the author counts.
		limit := rate
		if c.HasRole(membership.RoleGuest) {
			limit = guestLimit
		}
		if !limit.allow(string(c.Subject)) {
			return pubsub.ValidationIgnore
		}
		return pubsub.ValidationAccept
	})
	if err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	if n.reports, err = n.ps.Join(topic); err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	sub, err := n.reports.Subscribe()
	if err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	n.wg.Go(func() {
		for {
			m, err := sub.Next(n.ctx)
			if err != nil {
				return
			}
			if n.spool != nil {
				if _, _, err := n.spool.Put(m.Data); err != nil {
					n.log.Warn("keeping a report", "err", err)
				}
			}
		}
	})
	if n.spool != nil {
		n.Handle(reportsSync, n.serveReports)
		n.wg.Go(func() { n.reportsLoop(rc.SyncInterval) })
	}
	return nil
}

// ReportsEnabled reports whether this node keeps and sends sealed reports.
func (n *Node) ReportsEnabled() bool { return n.spool != nil }

// CanReport reports whether this node may author reports: it keeps a spool
// and is either a member or a guest with a proof of work.
func (n *Node) CanReport() bool { return n.spool != nil && (n.cfg.Cert != nil || n.cfg.Proof != nil) }

// PublishReport seals data to the network's root key and spreads it. Only
// certified members and guests may; the content is readable by the root
// alone, who weighs a guest's word less than a member's.
func (n *Node) PublishReport(ctx context.Context, data []byte) (string, error) {
	switch {
	case n.spool == nil:
		return "", errors.New("reports are not enabled on this node")
	case n.cfg.Cert == nil && n.cfg.Proof == nil:
		return "", errors.New("only members with a certificate and guests with a proof of work can send reports")
	case len(data) == 0 || len(data) > MaxReport:
		return "", fmt.Errorf("a report is 1 to %d bytes", MaxReport)
	}
	root, err := rootEd25519(n.cfg.Network)
	if err != nil {
		return "", err
	}
	env, err := sealed.Seal(root, data, ReportContext(n.cfg.Network.ID))
	if err != nil {
		return "", err
	}
	now := time.Now().UnixMilli()
	sig, err := n.cfg.Key.Sign(packetDigest(now, env))
	if err != nil {
		return "", err
	}
	rp := ReportPacket{Version: 1, Cert: n.cfg.Cert, Time: now, Sig: sig, Sealed: env}
	if n.cfg.Cert == nil {
		pub, err := crypto.MarshalPublicKey(n.cfg.Key.GetPublic())
		if err != nil {
			return "", err
		}
		rp.Guest = &GuestAuthor{Key: pub, Proof: *n.cfg.Proof}
	}
	packet, err := json.Marshal(rp)
	if err != nil {
		return "", err
	}
	id, _, err := n.spool.Put(packet)
	if err != nil {
		return "", err
	}
	// Published even when only mesh neighbours later pull it: gossip is
	// best effort, the sync is what makes delivery certain.
	if err := n.reports.Publish(ctx, packet); err != nil && !errors.Is(err, pubsub.ErrTopicClosed) {
		n.log.Debug("report gossip", "err", err)
	}
	return id, nil
}

// Reports returns up to limit sealed packets that arrived after the cursor,
// oldest first, no more than maxBytes together.
func (n *Node) Reports(after int64, limit int, maxBytes int64) []Packet {
	if n.spool == nil {
		return nil
	}
	return n.spool.After(after, limit, maxBytes, nil)
}

// ---------- sync between members ----------

type pullRequest struct {
	Have []string `json:"have"`
}

func (n *Node) serveReports(s network.Stream, c *membership.Cert) {
	defer s.Close()
	if c.HasRole(membership.RoleGuest) {
		s.Reset()
		return
	}
	s.SetDeadline(time.Now().Add(time.Minute))
	var req pullRequest
	line, err := bufio.NewReaderSize(io.LimitReader(s, 4*pullHaveLimit*40), 1<<16).ReadBytes('\n')
	if err != nil || json.Unmarshal(line, &req) != nil || len(req.Have) > pullHaveLimit {
		s.Reset()
		return
	}
	have := make(map[string]bool, len(req.Have))
	for _, id := range req.Have {
		have[id] = true
	}
	for _, p := range n.spool.After(0, pullMaxCount, pullMaxBytes, have) {
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(p.Data)))
		if _, err := s.Write(append(hdr[:], p.Data...)); err != nil {
			return
		}
	}
}

func (n *Node) reportsLoop(every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-n.ctx.Done():
			return
		case <-t.C:
		}
		var peers []peer.ID
		for _, m := range n.Members() {
			// A guest asks members only; a member also asks guests, who
			// have no other way to hand over their own reports.
			if m.Connected && (!m.Guest || n.cfg.Cert != nil) {
				peers = append(peers, m.ID)
			}
		}
		if len(peers) == 0 {
			continue
		}
		p := peers[rand.IntN(len(peers))]
		ctx, cancel := context.WithTimeout(n.ctx, time.Minute)
		got, err := n.pullReports(ctx, p)
		cancel()
		if err != nil {
			n.log.Debug("report sync", "peer", p.ShortString(), "err", err)
		} else if got > 0 {
			n.log.Debug("report sync", "peer", p.ShortString(), "new", got)
		}
	}
}

func (n *Node) pullReports(ctx context.Context, p peer.ID) (int, error) {
	s, err := n.NewStream(ctx, p, reportsSync)
	if err != nil {
		return 0, err
	}
	defer s.Close()
	s.SetDeadline(time.Now().Add(time.Minute))
	req, _ := json.Marshal(pullRequest{Have: n.spool.IDs(pullHaveLimit)})
	if _, err := s.Write(append(req, '\n')); err != nil {
		return 0, err
	}
	s.CloseWrite()
	r := bufio.NewReader(s)
	added := 0
	for i := 0; i < pullMaxCount; i++ {
		var hdr [4]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return added, err
		}
		size := binary.BigEndian.Uint32(hdr[:])
		if size == 0 || size > maxPacket {
			return added, errors.New("bad packet size")
		}
		buf := make([]byte, size)
		if _, err := io.ReadFull(r, buf); err != nil {
			return added, err
		}
		// A peer's word is not enough: the packet must stand on its own.
		if _, _, err := n.verifyReport(buf); err != nil {
			return added, fmt.Errorf("peer sent an invalid report: %w", err)
		}
		if _, ok, err := n.spool.Put(buf); err != nil {
			return added, err
		} else if ok {
			added++
		}
	}
	return added, nil
}

// rateLimit allows count events per window per key, approximately.
type rateLimit struct {
	count  int
	window time.Duration
	mu     sync.Mutex
	seen   map[string][]time.Time
}

func newRateLimit(count int, window time.Duration) *rateLimit {
	return &rateLimit{count: count, window: window, seen: map[string][]time.Time{}}
}

func (r *rateLimit) allow(key string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	list := r.seen[key][:0]
	for _, t := range r.seen[key] {
		if now.Sub(t) < r.window {
			list = append(list, t)
		}
	}
	if len(list) >= r.count {
		r.seen[key] = list
		return false
	}
	r.seen[key] = append(list, now)
	if len(r.seen) > 4096 { // forget idle authors
		for k, v := range r.seen {
			if len(v) == 0 || now.Sub(v[len(v)-1]) >= r.window {
				delete(r.seen, k)
			}
		}
	}
	return true
}
