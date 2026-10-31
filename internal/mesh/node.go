// Package mesh is the Rakkan network layer: a libp2p host admitting only
// members of a network, with discovery, failure detection and recovery.
// When the root opens the network (membership.Policy), it also admits
// guests proving work on their identity; Member.Guest and IsCertified tell
// them apart, so services can trust members and verify guests.
//
// Admission: right after a connection is established, both sides exchange
// certificates (see package membership) over the auth protocol. Peers that
// do not authenticate within AuthTimeout are disconnected and briefly
// denied by the connection gater. Every application protocol (the DHT,
// pubsub, and protocols registered with Handle) is gated: streams from or
// to peers that have not authenticated are refused, so non-members cannot
// even query the DHT for member addresses.
//
// Discovery: configured bootstrap peers, mDNS on the local network, and a
// private Kademlia DHT whose routing table holds members only, on which
// members advertise a rendezvous record for the network.
//
// Recovery: members are pinged periodically; connections that stop
// answering are closed. Known members that disconnect are redialled with
// exponential backoff, looking up fresh addresses in the DHT. NAT traversal
// uses UPnP, hole punching, and relays run by members.
//
// Revocation: revocation lists signed by the network root spread by gossip
// and in every auth exchange; revoked members are disconnected and denied.
package mesh

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/libp2p/go-libp2p/p2p/host/autorelay"
	"github.com/libp2p/go-libp2p/p2p/net/connmgr"
	circuitproto "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/proto"
	relayv2 "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	ma "github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

// Config configures a Node.
type Config struct {
	Key     crypto.PrivKey
	Network *membership.Network
	// Cert makes the node a member. A node without one joins an open
	// network as a guest with Proof, which must satisfy the policy.
	Cert  *membership.SignedCert
	Proof *membership.Proof
	// OnProof is called with a new proof the node solved because the
	// policy raised the difficulty, so it can be saved.
	OnProof func(membership.Proof)
	// CRL is the newest revocation list known at start, if any.
	CRL *membership.SignedCRL
	// Policy is the newest network policy known at start, if any; without
	// one the network is closed until a policy arrives from the network.
	Policy *membership.SignedPolicy
	// MaxGuests bounds guests connected at once. Default 256.
	MaxGuests int

	// ListenAddrs default to QUIC and TCP on all interfaces, random ports.
	ListenAddrs []string
	Bootstrap   []peer.AddrInfo
	// PeerCache is a file remembering members' addresses across restarts,
	// so a node can rejoin when no bootstrap peer is up. Empty disables it.
	PeerCache string
	// MDNS enables discovery on the local network.
	MDNS bool
	// NAT enables UPnP port mapping, hole punching and AutoNAT.
	NAT bool
	// RelayService lets this node relay traffic between members that cannot
	// reach each other directly. Enable it on publicly reachable nodes.
	RelayService bool
	// Reachability says whether other nodes can dial this one. The default,
	// ReachabilityAuto, decides from interface addresses (see reachability).
	Reachability Reachability

	AuthTimeout       time.Duration // default 10s
	HealthInterval    time.Duration // default 15s
	DiscoveryInterval time.Duration // default 1m
	MaxBackoff        time.Duration // default 5m
	Logger            *slog.Logger
}

func (c *Config) setDefaults() {
	if len(c.ListenAddrs) == 0 {
		c.ListenAddrs = []string{
			"/ip4/0.0.0.0/udp/0/quic-v1", "/ip6/::/udp/0/quic-v1",
			"/ip4/0.0.0.0/tcp/0", "/ip6/::/tcp/0",
		}
	}
	if c.AuthTimeout <= 0 {
		c.AuthTimeout = 10 * time.Second
	}
	if c.HealthInterval <= 0 {
		c.HealthInterval = 15 * time.Second
	}
	if c.DiscoveryInterval <= 0 {
		c.DiscoveryInterval = time.Minute
	}
	if c.MaxBackoff <= 0 {
		c.MaxBackoff = 5 * time.Minute
	}
	if c.MaxGuests <= 0 {
		c.MaxGuests = 256
	}
	if c.Logger == nil {
		c.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
}

// Reachability is whether a node accepts inbound connections from the
// internet.
type Reachability int

const (
	ReachabilityAuto Reachability = iota
	ReachabilityPublic
	ReachabilityPrivate
)

// ParseReachability parses "auto", "public" or "private".
func ParseReachability(s string) (Reachability, error) {
	switch s {
	case "auto", "":
		return ReachabilityAuto, nil
	case "public":
		return ReachabilityPublic, nil
	case "private":
		return ReachabilityPrivate, nil
	}
	return 0, fmt.Errorf("reachability %q: want auto, public or private", s)
}

// Node is a member of a Rakkan network.
type Node struct {
	cfg      Config
	log      *slog.Logger
	host     host.Host
	gated    *gatedHost
	dht      *dht.IpfsDHT
	ps       *pubsub.PubSub
	crl      *pubsub.Topic
	policy   *pubsub.Topic
	mdns     mdns.Service
	relay    *relayv2.Relay
	revs     *membership.Revocations
	policies *membership.Policies
	self     *membership.Cert

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	discover chan struct{} // requests a discovery round

	mu       sync.Mutex
	peers    map[peer.ID]*peerState
	observed map[peer.ID]ma.Multiaddr // our address as members see it
	denied   map[peer.ID]time.Time
	authing  map[peer.ID]bool
	proof    membership.Proof // a guest's own
}

// New starts a node.
func New(cfg Config) (*Node, error) {
	cfg.setDefaults()
	if cfg.Key == nil || cfg.Network == nil || (cfg.Cert == nil && cfg.Proof == nil) {
		return nil, errors.New("mesh: key, network, and a certificate or proof of work are required")
	}
	id, err := peer.IDFromPrivateKey(cfg.Key)
	if err != nil {
		return nil, err
	}
	n := &Node{
		cfg:      cfg,
		log:      cfg.Logger.With("node", id.ShortString()),
		revs:     membership.NewRevocations(cfg.Network),
		policies: membership.NewPolicies(cfg.Network),
		peers:    map[peer.ID]*peerState{},
		denied:   map[peer.ID]time.Time{},
		authing:  map[peer.ID]bool{},
		observed: map[peer.ID]ma.Multiaddr{},
		discover: make(chan struct{}, 1),
	}
	if cfg.CRL != nil {
		if _, err := n.revs.Update(cfg.CRL); err != nil {
			return nil, fmt.Errorf("mesh: %w", err)
		}
	}
	if cfg.Policy != nil {
		if _, err := n.policies.Update(cfg.Policy); err != nil {
			return nil, fmt.Errorf("mesh: %w", err)
		}
	}
	if cfg.Cert != nil {
		if n.self, err = cfg.Network.Verify(cfg.Cert, time.Now(), n.revs.Revoked); err != nil {
			return nil, fmt.Errorf("mesh: own certificate: %w", err)
		}
		if n.self.Subject != id {
			return nil, fmt.Errorf("mesh: certificate is for %s, but the key is %s", n.self.Subject, id)
		}
	} else {
		// Peers judge the proof against the newest policy they know; this
		// node's may be older, so it is not checked here.
		n.proof = *cfg.Proof
		n.self = &membership.Cert{Version: 1, Network: cfg.Network.ID, Subject: id, Roles: []string{membership.RoleGuest}}
	}
	n.ctx, n.cancel = context.WithCancel(context.Background())

	cm, err := connmgr.NewConnManager(64, 256, connmgr.WithGracePeriod(time.Minute))
	if err != nil {
		return nil, err
	}
	opts := []libp2p.Option{
		libp2p.Identity(cfg.Key),
		libp2p.ListenAddrStrings(cfg.ListenAddrs...),
		libp2p.ConnectionGater(&gater{n}),
		libp2p.ConnectionManager(cm),
		libp2p.UserAgent("rakkan/0.2"),
		libp2p.AddrsFactory(n.advertisedAddrs),
		libp2p.EnableRelay(),
		// Defaults suit large public networks: wait for 4 candidates and up
		// to 3 minutes, and ask for candidates every 30s. A closed network
		// may have a single relay, known only once it has authenticated.
		libp2p.EnableAutoRelayWithPeerSource(n.relayCandidates,
			autorelay.WithMinCandidates(1), autorelay.WithBootDelay(0),
			autorelay.WithMinInterval(5*time.Second)),
		// libp2p.EnableRelayService would wait for AutoNAT to confirm the
		// node is public; the relay is started directly below instead.
	}
	if cfg.NAT {
		opts = append(opts, libp2p.NATPortMap(), libp2p.EnableHolePunching(), libp2p.EnableNATService())
	}
	switch reach := n.reachability(); reach {
	case ReachabilityPublic:
		opts = append(opts, libp2p.ForceReachabilityPublic())
	case ReachabilityPrivate:
		opts = append(opts, libp2p.ForceReachabilityPrivate())
	}
	if n.host, err = libp2p.New(opts...); err != nil {
		n.cancel()
		return nil, fmt.Errorf("mesh: %w", err)
	}
	if cfg.RelayService {
		// The operator asked for a relay; do not wait for AutoNAT, which in
		// a small network may never gather enough observations.
		if n.relay, err = relayv2.New(n.host, relayv2.WithACL(relayACL{n}), relayv2.WithInfiniteLimits()); err != nil {
			n.host.Close()
			n.cancel()
			return nil, fmt.Errorf("mesh: relay: %w", err)
		}
	}
	n.gated = &gatedHost{Host: n.host, n: n}
	n.host.SetStreamHandler(authProtocol, n.handleAuth)
	n.host.SetStreamHandler(policyProtocol, n.handlePolicy)
	n.host.Network().Notify(&network.NotifyBundle{
		ConnectedF:    func(_ network.Network, c network.Conn) { n.onConnected(c) },
		DisconnectedF: func(_ network.Network, c network.Conn) { n.onDisconnected(c) },
	})

	if err := n.startServices(); err != nil {
		n.Close()
		return nil, err
	}
	n.log.Info("node started", "addrs", n.host.Addrs(), "network", cfg.Network.ID)
	return n, nil
}

func (n *Node) startServices() error {
	var err error
	n.dht, err = dht.New(n.gated,
		dht.Mode(dht.ModeAutoServer),
		dht.ProtocolPrefix(protocol.ID("/rakkan/"+n.tag())),
		dht.RoutingTableFilter(func(_ any, p peer.ID) bool { return n.isMember(p) }),
		dht.BootstrapPeers(n.cfg.Bootstrap...),
	)
	if err != nil {
		return fmt.Errorf("mesh: dht: %w", err)
	}
	n.ps, err = pubsub.NewGossipSub(n.ctx, n.gated,
		pubsub.WithPeerFilter(func(p peer.ID, _ string) bool { return n.isMember(p) }),
		pubsub.WithMessageSignaturePolicy(pubsub.StrictSign),
	)
	if err != nil {
		return fmt.Errorf("mesh: pubsub: %w", err)
	}
	if err := n.startCRLGossip(); err != nil {
		return err
	}
	if err := n.startPolicyGossip(); err != nil {
		return err
	}
	if n.cfg.MDNS {
		n.mdns = mdns.NewMdnsService(n.host, "rakkan-"+n.tag(), mdnsNotifee{n})
		if err := n.mdns.Start(); err != nil {
			return fmt.Errorf("mesh: mdns: %w", err)
		}
	}
	for _, ai := range n.cfg.Bootstrap {
		n.addDialTarget(ai)
	}
	n.loadPeerCache()
	n.wg.Go(n.healthLoop)
	n.wg.Go(n.dialLoop)
	n.wg.Go(n.discoveryLoop)
	return nil
}

// Close shuts the node down.
func (n *Node) Close() error {
	if n.host != nil {
		n.savePeerCache()
	}
	n.cancel()
	if n.relay != nil {
		n.relay.Close()
	}
	if n.mdns != nil {
		n.mdns.Close()
	}
	if n.dht != nil {
		n.dht.Close()
	}
	var err error
	if n.host != nil {
		err = n.host.Close()
	}
	n.wg.Wait()
	return err
}

// ID returns the node's peer ID.
func (n *Node) ID() peer.ID { return n.host.ID() }

// AddrInfo returns how other nodes can reach this one.
func (n *Node) AddrInfo() peer.AddrInfo {
	return peer.AddrInfo{ID: n.host.ID(), Addrs: n.host.Addrs()}
}

// Handle registers a handler for an application protocol. Only
// authenticated peers reach it; their certificate is passed along, which
// for a guest is synthesized and carries membership.RoleGuest.
func (n *Node) Handle(pid protocol.ID, h func(network.Stream, *membership.Cert)) {
	n.gated.SetStreamHandler(pid, func(s network.Stream) {
		c := n.memberCert(s.Conn().RemotePeer())
		if c == nil {
			s.Reset()
			return
		}
		h(s, c)
	})
}

// NewStream opens an application protocol stream to a member, connecting
// first if needed.
func (n *Node) NewStream(ctx context.Context, p peer.ID, pid protocol.ID) (network.Stream, error) {
	return n.gated.NewStream(ctx, p, pid)
}

// IsMember reports whether p is currently authenticated, as a member or a
// guest.
func (n *Node) IsMember(p peer.ID) bool { return n.isMember(p) }

// IsCertified reports whether p is currently authenticated with a
// certificate: a member, not a guest.
func (n *Node) IsCertified(p peer.ID) bool { return n.isCertified(p) }

// Certified reports whether this node holds a certificate.
func (n *Node) Certified() bool { return n.cfg.Cert != nil }

// JoinTopic joins a gossip topic scoped to this network. Only members take
// part, and messages are signed by their authors.
func (n *Node) JoinTopic(name string) (*pubsub.Topic, error) {
	return n.ps.Join("/rakkan/" + n.tag() + "/" + name)
}

// Connect dials a peer; it becomes usable once it authenticates.
func (n *Node) Connect(ctx context.Context, ai peer.AddrInfo) error {
	n.addDialTarget(ai)
	if err := n.host.Connect(ctx, ai); err != nil {
		return err
	}
	return n.waitMember(ctx, ai.ID)
}

// reachability resolves ReachabilityAuto. AutoNAT cannot be relied on in a
// small network: it needs several peers to dial back, and a single peer
// dialling back over the existing 5-tuple passes through a NAT and reports
// a NATed node as public. Instead: a node listening on a public address is
// public; one listening only on private addresses is behind NAT
// and uses relays and hole punching. A port mapped by UPnP is still
// advertised and dialled directly; the node is merely also given a relay
// address. Loopback-only nodes (tests, local clusters) are left to AutoNAT.
func (n *Node) reachability() Reachability {
	if n.cfg.Reachability != ReachabilityAuto {
		return n.cfg.Reachability
	}
	// The addresses the node listens on, with wildcards expanded.
	var addrs []ma.Multiaddr
	ifaces, err := manet.InterfaceMultiaddrs()
	if err != nil {
		return ReachabilityAuto
	}
	for _, s := range n.cfg.ListenAddrs {
		l, err := ma.NewMultiaddr(s)
		if err != nil {
			continue
		}
		if expanded, err := manet.ResolveUnspecifiedAddress(l, ifaces); err == nil {
			addrs = append(addrs, expanded...)
		}
	}
	sawPrivate := false
	for _, a := range addrs {
		switch {
		case manet.IsIPLoopback(a):
		case manet.IsPublicAddr(a):
			return ReachabilityPublic
		default:
			sawPrivate = true
		}
	}
	if sawPrivate {
		return ReachabilityPrivate
	}
	return ReachabilityAuto
}

// advertisedAddrs is what the node tells others: its listen addresses
// without loopback, plus the external addresses members observed. libp2p's
// own observed-address tracking needs four distinct observers, which a
// small network may never have, and counts any connected peer; this counts
// authenticated members only, and one is enough. Hole punching needs it:
// without a known external address there is nothing to punch.
func (n *Node) advertisedAddrs(addrs []ma.Multiaddr) []ma.Multiaddr {
	out := withoutLoopback(addrs)
	n.mu.Lock()
	defer n.mu.Unlock()
	for _, a := range n.observed {
		if !slices.ContainsFunc(out, a.Equal) {
			out = append(out, a)
		}
	}
	return out
}

// withoutLoopback stops a node from advertising loopback addresses, which
// make other nodes dial themselves, unless it has nothing else.
func withoutLoopback(addrs []ma.Multiaddr) []ma.Multiaddr {
	var out []ma.Multiaddr
	for _, a := range addrs {
		if !manet.IsIPLoopback(a) {
			out = append(out, a)
		}
	}
	if len(out) == 0 {
		return addrs
	}
	return out
}

// tag is a short network identifier used in protocol and service names.
func (n *Node) tag() string {
	id := n.cfg.Network.ID
	return id[len(id)-12:]
}

type mdnsNotifee struct{ n *Node }

func (m mdnsNotifee) HandlePeerFound(ai peer.AddrInfo) {
	if ai.ID == m.n.host.ID() {
		return
	}
	m.n.log.Debug("found peer via mDNS", "peer", ai.ID.ShortString())
	m.n.addDialTarget(ai)
}

// relayACL admits members to this node's relay service: they may reserve
// a slot and be reached through it. Guests may reach members through it,
// not hold slots, so a relay's bandwidth serves the members its operator
// vouched for.
type relayACL struct{ n *Node }

func (r relayACL) AllowReserve(p peer.ID, _ ma.Multiaddr) bool { return r.n.isCertified(p) }

func (r relayACL) AllowConnect(src peer.ID, _ ma.Multiaddr, dest peer.ID) bool {
	return r.n.isMember(src) && r.n.isCertified(dest)
}

// relayCandidates offers connected members running a relay to AutoRelay.
func (n *Node) relayCandidates(ctx context.Context, num int) <-chan peer.AddrInfo {
	out := make(chan peer.AddrInfo, num)
	go func() {
		defer close(out)
		for _, m := range n.Members() {
			if !m.Connected || num == 0 {
				continue
			}
			if ok, _ := n.host.Peerstore().SupportsProtocols(m.ID, circuitproto.ProtoIDv2Hop); len(ok) == 0 {
				continue
			}
			select {
			case out <- peer.AddrInfo{ID: m.ID, Addrs: n.host.Peerstore().Addrs(m.ID)}:
				num--
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}
