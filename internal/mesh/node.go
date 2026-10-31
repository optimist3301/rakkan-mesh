// Package mesh is the Rakkan network layer: a libp2p host admitting only
// members of a closed network, with discovery, failure detection and
// recovery.
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
	"github.com/libp2p/go-libp2p/p2p/net/connmgr"
	relayv2 "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	ma "github.com/multiformats/go-multiaddr"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

// Config configures a Node.
type Config struct {
	Key     crypto.PrivKey
	Network *membership.Network
	Cert    *membership.SignedCert
	// CRL is the newest revocation list known at start, if any.
	CRL *membership.SignedCRL

	// ListenAddrs default to QUIC and TCP on all interfaces, random ports.
	ListenAddrs []string
	Bootstrap   []peer.AddrInfo
	// MDNS enables discovery on the local network.
	MDNS bool
	// NAT enables UPnP port mapping, hole punching and AutoNAT.
	NAT bool
	// RelayService lets this node relay traffic between members that cannot
	// reach each other directly. Enable it on publicly reachable nodes.
	RelayService bool

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
	if c.Logger == nil {
		c.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
}

// Node is a member of a Rakkan network.
type Node struct {
	cfg   Config
	log   *slog.Logger
	host  host.Host
	gated *gatedHost
	dht   *dht.IpfsDHT
	ps    *pubsub.PubSub
	crl   *pubsub.Topic
	mdns  mdns.Service
	revs  *membership.Revocations
	self  *membership.Cert

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	discover chan struct{} // requests a discovery round

	mu      sync.Mutex
	peers   map[peer.ID]*peerState
	denied  map[peer.ID]time.Time
	authing map[peer.ID]bool
}

// New starts a node.
func New(cfg Config) (*Node, error) {
	cfg.setDefaults()
	if cfg.Key == nil || cfg.Network == nil || cfg.Cert == nil {
		return nil, errors.New("mesh: key, network and certificate are required")
	}
	id, err := peer.IDFromPrivateKey(cfg.Key)
	if err != nil {
		return nil, err
	}
	n := &Node{
		cfg:      cfg,
		log:      cfg.Logger.With("node", id.ShortString()),
		revs:     membership.NewRevocations(cfg.Network),
		peers:    map[peer.ID]*peerState{},
		denied:   map[peer.ID]time.Time{},
		authing:  map[peer.ID]bool{},
		discover: make(chan struct{}, 1),
	}
	if cfg.CRL != nil {
		if _, err := n.revs.Update(cfg.CRL); err != nil {
			return nil, fmt.Errorf("mesh: %w", err)
		}
	}
	if n.self, err = cfg.Network.Verify(cfg.Cert, time.Now(), n.revs.Revoked); err != nil {
		return nil, fmt.Errorf("mesh: own certificate: %w", err)
	}
	if n.self.Subject != id {
		return nil, fmt.Errorf("mesh: certificate is for %s, but the key is %s", n.self.Subject, id)
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
		libp2p.EnableRelay(),
		libp2p.EnableAutoRelayWithPeerSource(n.relayCandidates),
	}
	if cfg.NAT {
		opts = append(opts, libp2p.NATPortMap(), libp2p.EnableHolePunching(), libp2p.EnableNATService())
	}
	if cfg.RelayService {
		opts = append(opts, libp2p.EnableRelayService(relayv2.WithACL(relayACL{n}), relayv2.WithInfiniteLimits()))
	}
	if n.host, err = libp2p.New(opts...); err != nil {
		n.cancel()
		return nil, fmt.Errorf("mesh: %w", err)
	}
	n.gated = &gatedHost{Host: n.host, n: n}
	n.host.SetStreamHandler(authProtocol, n.handleAuth)
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
	if n.cfg.MDNS {
		n.mdns = mdns.NewMdnsService(n.host, "rakkan-"+n.tag(), mdnsNotifee{n})
		if err := n.mdns.Start(); err != nil {
			return fmt.Errorf("mesh: mdns: %w", err)
		}
	}
	for _, ai := range n.cfg.Bootstrap {
		n.addDialTarget(ai)
	}
	n.wg.Go(n.healthLoop)
	n.wg.Go(n.dialLoop)
	n.wg.Go(n.discoveryLoop)
	return nil
}

// Close shuts the node down.
func (n *Node) Close() error {
	n.cancel()
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
// authenticated members reach it; their certificate is passed along.
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

// Connect dials a peer; it becomes usable once it authenticates.
func (n *Node) Connect(ctx context.Context, ai peer.AddrInfo) error {
	n.addDialTarget(ai)
	if err := n.host.Connect(ctx, ai); err != nil {
		return err
	}
	return n.waitMember(ctx, ai.ID)
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

// relayACL admits only members to this node's relay service.
type relayACL struct{ n *Node }

func (r relayACL) AllowReserve(p peer.ID, _ ma.Multiaddr) bool { return r.n.isMember(p) }

func (r relayACL) AllowConnect(src peer.ID, _ ma.Multiaddr, dest peer.ID) bool {
	return r.n.isMember(src) && r.n.isMember(dest)
}

// relayCandidates offers connected members as relays to AutoRelay.
func (n *Node) relayCandidates(ctx context.Context, num int) <-chan peer.AddrInfo {
	out := make(chan peer.AddrInfo, num)
	go func() {
		defer close(out)
		for _, m := range n.Members() {
			if !m.Connected || num == 0 {
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
