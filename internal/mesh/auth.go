package mesh

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/libp2p/go-libp2p/core/control"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	ma "github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

const authProtocol = protocol.ID("/rakkan/auth/1.0.0")

// maxHello bounds an auth message: a certificate chain and a revocation list.
const maxHello = 256 << 10

// denyAfterFailure is how long a peer that failed to authenticate is refused.
const denyAfterFailure = time.Minute

var errNotMember = errors.New("peer is not an authenticated member")

type hello struct {
	// Cert is a member's certificate; a guest sends Proof instead.
	Cert   *membership.SignedCert   `json:"cert,omitempty"`
	Proof  *membership.Proof        `json:"proof,omitempty"`
	CRL    *membership.SignedCRL    `json:"crl,omitempty"`
	Policy *membership.SignedPolicy `json:"policy,omitempty"`
	// Observed is the address the sender sees the receiver connect from,
	// on direct connections. A node behind NAT learns its external address
	// this way, from members only (see advertisedAddrs).
	Observed string `json:"observed,omitempty"`
}

// peerState is what a node knows about a peer it has connected to.
type peerState struct {
	cert   *membership.Cert // set once authenticated, kept across disconnects
	signed *membership.SignedCert
	proof  *membership.Proof // a guest's, instead of signed
	authed bool              // authenticated on a live connection
	ready  chan struct{}     // closed when authed becomes true

	rtt      time.Duration
	lastSeen time.Time
	failures int // consecutive failed pings

	dial     bool // redial when disconnected
	nextDial time.Time
	backoff  time.Duration
}

func (n *Node) state(p peer.ID) *peerState {
	st, ok := n.peers[p]
	if !ok {
		st = &peerState{ready: make(chan struct{})}
		n.peers[p] = st
	}
	return st
}

func (n *Node) isMember(p peer.ID) bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	st, ok := n.peers[p]
	return ok && st.authed
}

// isCertified reports whether p is an authenticated member holding a
// certificate, as opposed to a guest.
func (n *Node) isCertified(p peer.ID) bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	st, ok := n.peers[p]
	return ok && st.authed && st.signed != nil
}

func (n *Node) memberCert(p peer.ID) *membership.Cert {
	n.mu.Lock()
	defer n.mu.Unlock()
	if st, ok := n.peers[p]; ok && st.authed {
		return st.cert
	}
	return nil
}

// waitMember blocks until p has authenticated or ctx ends.
func (n *Node) waitMember(ctx context.Context, p peer.ID) error {
	n.mu.Lock()
	st := n.state(p)
	ready, authed := st.ready, st.authed
	n.mu.Unlock()
	if authed {
		return nil
	}
	select {
	case <-ready:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("%w: %s", errNotMember, p.ShortString())
	}
}

func (n *Node) isDenied(p peer.ID) bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	until, ok := n.denied[p]
	if ok && time.Now().After(until) {
		delete(n.denied, p)
		return false
	}
	return ok
}

// deny refuses connections from p for d and drops existing ones.
func (n *Node) deny(p peer.ID, d time.Duration, reason error) {
	n.mu.Lock()
	n.denied[p] = time.Now().Add(d)
	n.mu.Unlock()
	n.log.Warn("denying peer", "peer", p.ShortString(), "for", d, "reason", reason)
	n.host.Network().ClosePeer(p)
}

// Ban refuses p for d and drops its connections; a member is redialled
// only after d.
func (n *Node) Ban(p peer.ID, d time.Duration, reason error) {
	n.mu.Lock()
	if st, ok := n.peers[p]; ok {
		st.nextDial = time.Now().Add(d)
	}
	n.mu.Unlock()
	n.deny(p, d, reason)
}

func (n *Node) onConnected(c network.Conn) {
	p := c.RemotePeer()
	n.log.Debug("connected", "peer", p.ShortString(), "addr", c.RemoteMultiaddr(), "dir", c.Stat().Direction)
	if n.isMember(p) {
		return
	}
	go n.authenticate(p)
	// Disconnect peers that never authenticate.
	time.AfterFunc(n.cfg.AuthTimeout, func() {
		if !n.isMember(p) && n.host.Network().Connectedness(p) == network.Connected {
			n.deny(p, denyAfterFailure, errors.New("authentication timed out"))
		}
	})
}

func (n *Node) onDisconnected(c network.Conn) {
	p := c.RemotePeer()
	if n.host.Network().Connectedness(p) == network.Connected {
		return // other connections remain
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	st, ok := n.peers[p]
	if !ok {
		return
	}
	if st.cert == nil && !st.dial {
		delete(n.peers, p) // a stranger: forget it
		return
	}
	if !st.authed {
		return
	}
	// The NAT mapping it saw may not outlive the connection.
	delete(n.observed, p)
	st.authed = false
	st.ready = make(chan struct{})
	st.failures = 0
	st.nextDial = time.Now()
	n.log.Info("member disconnected", "peer", p.ShortString())
}

// authenticate runs the auth protocol as the initiator. Both sides usually
// initiate; either exchange suffices.
func (n *Node) authenticate(p peer.ID) {
	n.mu.Lock()
	if n.authing[p] {
		n.mu.Unlock()
		return
	}
	n.authing[p] = true
	n.mu.Unlock()
	defer func() {
		n.mu.Lock()
		delete(n.authing, p)
		n.mu.Unlock()
	}()

	ctx, cancel := context.WithTimeout(n.ctx, n.cfg.AuthTimeout)
	defer cancel()
	s, err := n.host.NewStream(network.WithAllowLimitedConn(ctx, "auth"), p, authProtocol)
	if err != nil {
		// The peer may still authenticate through its own exchange.
		n.log.Debug("auth: open stream", "peer", p.ShortString(), "err", err)
		return
	}
	defer s.Close()
	s.SetDeadline(time.Now().Add(n.cfg.AuthTimeout))
	if err := n.writeHello(s); err != nil {
		s.Reset()
		return
	}
	n.readHello(s, p)
}

func (n *Node) handleAuth(s network.Stream) {
	defer s.Close()
	s.SetDeadline(time.Now().Add(n.cfg.AuthTimeout))
	if !n.readHello(s, s.Conn().RemotePeer()) {
		s.Reset()
		return
	}
	n.writeHello(s)
}

func (n *Node) writeHello(s network.Stream) error {
	crl, _ := n.revs.Current()
	policy, _ := n.policies.Current()
	h := hello{Cert: n.cfg.Cert, CRL: crl, Policy: policy}
	if h.Cert == nil {
		n.mu.Lock()
		pr := n.proof
		n.mu.Unlock()
		h.Proof = &pr
	}
	if !isRelayed(s.Conn()) {
		h.Observed = s.Conn().RemoteMultiaddr().String()
	}
	b, err := json.Marshal(h)
	if err != nil {
		return err
	}
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := s.Write(hdr[:]); err != nil {
		return err
	}
	_, err = s.Write(b)
	return err
}

// readHello reads and verifies a peer's hello, admitting it on success.
func (n *Node) readHello(r network.Stream, p peer.ID) bool {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		n.log.Debug("auth: read", "peer", p.ShortString(), "err", err)
		return false
	}
	size := binary.BigEndian.Uint32(hdr[:])
	if size > maxHello {
		n.deny(p, denyAfterFailure, fmt.Errorf("auth message of %d bytes", size))
		return false
	}
	b := make([]byte, size)
	if _, err := io.ReadFull(r, b); err != nil {
		n.log.Debug("auth: read", "peer", p.ShortString(), "err", err)
		return false
	}
	var h hello
	if err := json.Unmarshal(b, &h); err != nil {
		n.deny(p, denyAfterFailure, err)
		return false
	}
	// A newer revocation list may revoke the peer itself, and a newer
	// policy may admit or exclude guests, so apply them first. Only the
	// root's signatures make them count.
	if h.CRL != nil {
		n.applyCRL(h.CRL)
	}
	if h.Policy != nil {
		n.applyPolicy(h.Policy)
	}
	var cert *membership.Cert
	var err error
	switch {
	case h.Cert != nil:
		cert, err = n.cfg.Network.Verify(h.Cert, time.Now(), n.revs.Revoked)
		if err == nil && cert.Subject != p {
			err = fmt.Errorf("certificate is for %s", cert.Subject)
		}
	case h.Proof != nil:
		cert, err = n.checkGuest(p, *h.Proof)
	default:
		err = errors.New("neither a certificate nor a proof of work")
	}
	if err != nil {
		d := denyAfterFailure
		if errors.Is(err, membership.ErrRevoked) {
			d = 100 * 365 * 24 * time.Hour
		}
		n.deny(p, d, err)
		return false
	}
	if h.Cert == nil {
		n.admitGuest(p, cert, *h.Proof)
		return true
	}
	n.admit(p, cert, h.Cert, nil)
	// A guest could report any address; members only.
	n.observe(p, h.Observed)
	return true
}

// errClosed refuses guests while the network's policy does not admit them.
var errClosed = errors.New("the network does not admit guests")

// checkGuest checks a guest's proof of work against the current policy.
// A guest's certificate is synthesized: it carries the guest role and no
// issuer or validity, and lasts as long as the policy admits the proof.
func (n *Node) checkGuest(p peer.ID, pr membership.Proof) (*membership.Cert, error) {
	_, pol := n.policies.Current()
	switch {
	case !pol.Open:
		return nil, errClosed
	case n.revs.Revoked(p.String()):
		return nil, fmt.Errorf("%w: peer %s", membership.ErrRevoked, p)
	}
	if w := n.cfg.Network.Work(p, pr); w < pol.PowBits {
		return nil, fmt.Errorf("proof of work of %d bits, the network wants %d", w, pol.PowBits)
	}
	return &membership.Cert{Version: 1, Network: n.cfg.Network.ID, Subject: p, Roles: []string{membership.RoleGuest}}, nil
}

// admitGuest admits a guest unless MaxGuests are already connected.
func (n *Node) admitGuest(p peer.ID, cert *membership.Cert, pr membership.Proof) {
	n.mu.Lock()
	guests := 0
	for q, st := range n.peers {
		if st.authed && st.signed == nil && q != p {
			guests++
		}
	}
	n.mu.Unlock()
	if guests >= n.cfg.MaxGuests {
		n.deny(p, denyAfterFailure, fmt.Errorf("%d guests already connected", guests))
		return
	}
	n.admit(p, cert, nil, &pr)
}

// recheck reports why an admitted peer may no longer stay: its certificate
// expired or was revoked, or the policy stopped admitting its proof.
func (n *Node) recheck(st *peerState, p peer.ID) error {
	if st.signed != nil {
		_, err := n.cfg.Network.Verify(st.signed, time.Now(), n.revs.Revoked)
		return err
	}
	if st.proof != nil {
		_, err := n.checkGuest(p, *st.proof)
		return err
	}
	return nil
}

// observe records the external address a member reports for this node.
// Only public addresses of transports the node listens on are kept.
func (n *Node) observe(p peer.ID, addr string) {
	if addr == "" {
		return
	}
	a, err := ma.NewMultiaddr(addr)
	if err != nil || !manet.IsPublicAddr(a) {
		return
	}
	if _, err := a.ValueForProtocol(ma.P_TCP); err != nil {
		if _, err := a.ValueForProtocol(ma.P_QUIC_V1); err != nil {
			return
		}
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if _, seen := n.observed[p]; !seen {
		n.log.Debug("member observed our address", "peer", p.ShortString(), "addr", a)
	}
	n.observed[p] = a
}

func (n *Node) admit(p peer.ID, cert *membership.Cert, signed *membership.SignedCert, proof *membership.Proof) {
	n.mu.Lock()
	st := n.state(p)
	wasAuthed := st.authed
	st.cert, st.signed, st.proof = cert, signed, proof
	st.authed, st.dial = true, true
	st.lastSeen, st.backoff, st.failures = time.Now(), 0, 0
	if !wasAuthed {
		close(st.ready)
	}
	n.mu.Unlock()
	if wasAuthed {
		return
	}
	weight := 100
	if signed == nil {
		weight = 10 // guests go first when connections are trimmed
	}
	n.host.ConnManager().TagPeer(p, "rakkan-member", weight)
	if n.dht != nil {
		n.dht.RoutingTable().TryAddPeer(p, true, false)
	}
	n.log.Info("member authenticated", "peer", p.ShortString(), "roles", cert.Roles)
	select {
	case n.discover <- struct{}{}:
	default:
	}
}

// gater refuses denied peers at dial and handshake time.
type gater struct{ n *Node }

func (g *gater) InterceptPeerDial(p peer.ID) bool             { return !g.n.isDenied(p) }
func (g *gater) InterceptAddrDial(peer.ID, ma.Multiaddr) bool { return true }
func (g *gater) InterceptAccept(network.ConnMultiaddrs) bool  { return true }
func (g *gater) InterceptUpgraded(network.Conn) (bool, control.DisconnectReason) {
	return true, 0
}
func (g *gater) InterceptSecured(_ network.Direction, p peer.ID, _ network.ConnMultiaddrs) bool {
	return !g.n.isDenied(p)
}

// gatedHost is the host as seen by the DHT, pubsub and application
// protocols: streams in either direction wait for the peer to authenticate
// and are refused if it does not within AuthTimeout.
type gatedHost struct {
	host.Host
	n *Node
}

func (g *gatedHost) SetStreamHandler(pid protocol.ID, h network.StreamHandler) {
	g.Host.SetStreamHandler(pid, g.gate(h))
}

func (g *gatedHost) SetStreamHandlerMatch(pid protocol.ID, m func(protocol.ID) bool, h network.StreamHandler) {
	g.Host.SetStreamHandlerMatch(pid, m, g.gate(h))
}

func (g *gatedHost) NewStream(ctx context.Context, p peer.ID, pids ...protocol.ID) (network.Stream, error) {
	wctx, cancel := context.WithTimeout(ctx, g.n.cfg.AuthTimeout)
	defer cancel()
	if g.n.host.Network().Connectedness(p) != network.Connected {
		if err := g.n.host.Connect(wctx, peer.AddrInfo{ID: p}); err != nil {
			return nil, err
		}
	}
	if err := g.n.waitMember(wctx, p); err != nil {
		return nil, err
	}
	return g.Host.NewStream(ctx, p, pids...)
}

func (g *gatedHost) gate(h network.StreamHandler) network.StreamHandler {
	return func(s network.Stream) {
		ctx, cancel := context.WithTimeout(g.n.ctx, g.n.cfg.AuthTimeout)
		defer cancel()
		if err := g.n.waitMember(ctx, s.Conn().RemotePeer()); err != nil {
			s.Reset()
			return
		}
		h(s)
	}
}
