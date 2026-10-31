package mesh

import (
	"bufio"
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net"
	"slices"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

const echoProtocol = protocol.ID("/rakkan/test/echo/1")

type testNet struct {
	root crypto.PrivKey
	net  *membership.Network
}

func newTestNet(t *testing.T) *testNet {
	t.Helper()
	root, n, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	return &testNet{root: root, net: n}
}

func newKey(t *testing.T) (crypto.PrivKey, peer.ID) {
	t.Helper()
	sk, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := peer.IDFromPrivateKey(sk)
	return sk, id
}

// issue returns a certificate for id, signed by the root or, if given, by
// an admin's key and certificate.
func (tn *testNet) issue(t *testing.T, id peer.ID, roles []string, admin crypto.PrivKey, adminCert *membership.SignedCert) *membership.SignedCert {
	t.Helper()
	signer := tn.root
	if admin != nil {
		signer = admin
	}
	c, err := membership.Issue(signer, adminCert, membership.Cert{
		Network: tn.net.ID, Subject: id, Roles: roles,
		NotBefore: certBase.Add(-time.Minute), NotAfter: certBase.Add(time.Hour),
	})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// certBase is shared so delegated certificates do not outlive their issuer's.
var certBase = time.Now().Truncate(time.Second)

type nodeOpt func(*Config)

func withBootstrap(nodes ...*Node) nodeOpt {
	return func(c *Config) {
		for _, n := range nodes {
			c.Bootstrap = append(c.Bootstrap, n.AddrInfo())
		}
	}
}

func withListen(addrs ...string) nodeOpt { return func(c *Config) { c.ListenAddrs = addrs } }

func (tn *testNet) config(t *testing.T, key crypto.PrivKey, cert *membership.SignedCert, opts ...nodeOpt) Config {
	cfg := Config{
		Key: key, Network: tn.net, Cert: cert,
		ListenAddrs:       []string{"/ip4/127.0.0.1/tcp/0", "/ip4/127.0.0.1/udp/0/quic-v1"},
		AuthTimeout:       2 * time.Second,
		HealthInterval:    300 * time.Millisecond,
		DiscoveryInterval: 500 * time.Millisecond,
		MaxBackoff:        time.Second,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

func start(t *testing.T, cfg Config) *Node {
	t.Helper()
	n, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { n.Close() })
	n.Handle(echoProtocol, func(s network.Stream, _ *membership.Cert) {
		defer s.Close()
		io.Copy(s, s)
	})
	return n
}

// member starts a node with a root-issued member certificate.
func (tn *testNet) member(t *testing.T, opts ...nodeOpt) *Node {
	t.Helper()
	key, id := newKey(t)
	return start(t, tn.config(t, key, tn.issue(t, id, []string{membership.RoleMember}, nil, nil), opts...))
}

func eventually(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s", timeout, what)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func connectedMember(n *Node, p peer.ID) bool {
	return slices.ContainsFunc(n.Members(), func(m Member) bool { return m.ID == p && m.Connected })
}

func echo(ctx context.Context, from *Node, to peer.ID, msg string) (string, error) {
	s, err := from.NewStream(ctx, to, echoProtocol)
	if err != nil {
		return "", err
	}
	defer s.Close()
	s.SetDeadline(time.Now().Add(5 * time.Second))
	if _, err := fmt.Fprintln(s, msg); err != nil {
		return "", err
	}
	s.CloseWrite()
	line, err := bufio.NewReader(s).ReadString('\n')
	return line, err
}

func TestMembersAuthenticateAndTalk(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t)
	b := tn.member(t, withBootstrap(a))
	eventually(t, 5*time.Second, "mutual membership", func() bool {
		return connectedMember(a, b.ID()) && connectedMember(b, a.ID())
	})
	got, err := echo(t.Context(), a, b.ID(), "ping")
	if err != nil || got != "ping\n" {
		t.Fatalf("echo: %q, %v", got, err)
	}
	eventually(t, 5*time.Second, "RTT measured", func() bool {
		return slices.ContainsFunc(a.Members(), func(m Member) bool { return m.ID == b.ID() && m.RTT > 0 })
	})
}

func TestForeignNetworkRejected(t *testing.T) {
	tn, other := newTestNet(t), newTestNet(t)
	a := tn.member(t)
	x := other.member(t, withBootstrap(a))

	time.Sleep(time.Second)
	if connectedMember(a, x.ID()) || connectedMember(x, a.ID()) {
		t.Fatal("member of another network admitted")
	}
	if _, err := echo(t.Context(), x, a.ID(), "hi"); err == nil {
		t.Fatal("application protocol reachable from another network")
	}
	eventually(t, 3*time.Second, "connection dropped", func() bool {
		return a.host.Network().Connectedness(x.ID()) != network.Connected
	})
}

func TestUncertifiedPeerIsGatedAndDropped(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t)
	h, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if err := h.Connect(t.Context(), a.AddrInfo()); err != nil {
		t.Fatal(err)
	}

	// Application and DHT protocols refuse the stranger.
	for _, pid := range []protocol.ID{echoProtocol, protocol.ID("/rakkan/" + a.tag() + "/kad/1.0.0")} {
		s, err := h.NewStream(t.Context(), a.ID(), pid)
		if err != nil {
			continue
		}
		s.SetDeadline(time.Now().Add(5 * time.Second))
		fmt.Fprintln(s, "hello")
		if _, err := s.Read(make([]byte, 1)); err == nil {
			t.Fatalf("%s answered an unauthenticated peer", pid)
		}
	}
	eventually(t, 4*time.Second, "stranger disconnected", func() bool {
		return h.Network().Connectedness(a.ID()) != network.Connected
	})
	// And it is denied for a while.
	if err := h.Connect(t.Context(), a.AddrInfo()); err == nil {
		time.Sleep(200 * time.Millisecond)
		if h.Network().Connectedness(a.ID()) == network.Connected {
			t.Fatal("denied peer reconnected")
		}
	}
}

func TestDiscoveryAndRelayFailure(t *testing.T) {
	tn := newTestNet(t)
	hub := tn.member(t)
	b := tn.member(t, withBootstrap(hub))
	c := tn.member(t, withBootstrap(hub))

	// b and c only know the hub; they must find each other through the DHT.
	eventually(t, 15*time.Second, "b and c discover each other", func() bool {
		return connectedMember(b, c.ID()) && connectedMember(c, b.ID())
	})
	// Losing the hub leaves them connected.
	hub.Close()
	time.Sleep(time.Second)
	if !connectedMember(b, c.ID()) {
		t.Fatal("b lost c when the hub went away")
	}
	if got, err := echo(t.Context(), b, c.ID(), "still here"); err != nil || got != "still here\n" {
		t.Fatalf("echo after hub failure: %q, %v", got, err)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func TestReconnectAfterRestart(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t)
	key, id := newKey(t)
	cert := tn.issue(t, id, []string{membership.RoleMember}, nil, nil)
	addr := fmt.Sprintf("/ip4/127.0.0.1/tcp/%d", freePort(t))

	b, err := New(tn.config(t, key, cert, withListen(addr), withBootstrap(a)))
	if err != nil {
		t.Fatal(err)
	}
	eventually(t, 5*time.Second, "initial membership", func() bool { return connectedMember(a, id) })

	b.Close()
	eventually(t, 5*time.Second, "disconnect noticed", func() bool { return !connectedMember(a, id) })

	// b comes back on the same address without knowing anyone: a must
	// redial it on its own.
	b, err = New(tn.config(t, key, cert, withListen(addr)))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	eventually(t, 10*time.Second, "a reconnects to b", func() bool {
		return connectedMember(a, id) && connectedMember(b, a.ID())
	})
}

func TestRevocationSpreads(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t)
	b := tn.member(t, withBootstrap(a))
	cKey, cID := newKey(t)
	cCert := tn.issue(t, cID, []string{membership.RoleMember}, nil, nil)
	c := start(t, tn.config(t, cKey, cCert, withBootstrap(a, b)))
	eventually(t, 10*time.Second, "full mesh", func() bool {
		return connectedMember(a, b.ID()) && connectedMember(a, cID) && connectedMember(b, cID)
	})
	// Give gossipsub a moment to form its mesh.
	time.Sleep(time.Second)

	cSerial, _ := tn.net.Verify(cCert, time.Now(), nil)
	crl, err := membership.IssueCRL(tn.root, membership.CRL{Network: tn.net.ID, Seq: 1, IssuedAt: time.Now(), Revoked: []string{cSerial.Serial}})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.PublishCRL(t.Context(), crl); err != nil {
		t.Fatal(err)
	}
	eventually(t, 5*time.Second, "revoked member dropped everywhere", func() bool {
		return !connectedMember(a, cID) && !connectedMember(b, cID) && b.isDenied(cID)
	})
	if !connectedMember(a, b.ID()) {
		t.Fatal("revocation disconnected an unrelated member")
	}
	_ = c
}

func TestDelegatedAdmission(t *testing.T) {
	tn := newTestNet(t)
	adminKey, adminID := newKey(t)
	adminCert := tn.issue(t, adminID, []string{membership.RoleMember, membership.RoleAdmin}, nil, nil)
	admin := start(t, tn.config(t, adminKey, adminCert))
	b := tn.member(t, withBootstrap(admin))

	// The admin admits a new node without the root key.
	dKey, dID := newKey(t)
	dCert := tn.issue(t, dID, []string{membership.RoleMember}, adminKey, adminCert)
	d := start(t, tn.config(t, dKey, dCert, withBootstrap(admin)))

	eventually(t, 15*time.Second, "delegated member accepted by everyone", func() bool {
		return connectedMember(admin, dID) && connectedMember(b, dID)
	})
	_ = d
}

func TestMDNSDiscovery(t *testing.T) {
	tn := newTestNet(t)
	// Listen on all interfaces: mDNS announces real interface addresses.
	listen := withListen("/ip4/0.0.0.0/tcp/0")
	mdnsOn := func(c *Config) { c.MDNS = true }
	a := tn.member(t, listen, mdnsOn)
	b := tn.member(t, listen, mdnsOn)
	deadline := time.Now().Add(10 * time.Second)
	for !(connectedMember(a, b.ID()) && connectedMember(b, a.ID())) {
		if time.Now().After(deadline) {
			t.Skip("no mDNS discovery within 10s; multicast may be unavailable here")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// A node restarting on a new address, with every bootstrap peer gone,
// rejoins through its peer cache alone: the others cannot redial it.
func TestPeerCacheRejoin(t *testing.T) {
	tn := newTestNet(t)
	cache := t.TempDir() + "/peers.json"
	a := tn.member(t)
	hub := tn.member(t)
	key, id := newKey(t)
	cert := tn.issue(t, id, []string{membership.RoleMember}, nil, nil)
	withCache := func(c *Config) { c.PeerCache = cache }

	b, err := New(tn.config(t, key, cert, withCache, withBootstrap(hub, a)))
	if err != nil {
		t.Fatal(err)
	}
	eventually(t, 10*time.Second, "b joins", func() bool { return connectedMember(a, id) })
	b.Close() // writes the cache
	hub.Close()

	b, err = New(tn.config(t, key, cert, withCache, withListen(fmt.Sprintf("/ip4/127.0.0.1/tcp/%d", freePort(t)))))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	eventually(t, 10*time.Second, "b rejoins from its cache", func() bool {
		return connectedMember(a, id) && connectedMember(b, a.ID())
	})
}
