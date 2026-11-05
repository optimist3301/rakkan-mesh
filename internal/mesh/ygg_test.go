package mesh

import (
	"fmt"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"

	"github.com/optimist3301/rakkan-mesh/internal/mldht"
)

// Two members whose only known address for each other is /ygg: the second
// is bootstrapped with nothing else, and their embedded Yggdrasil nodes
// peer over loopback, forming an overlay of their own.
func TestMembersConnectOverYggdrasil(t *testing.T) {
	tn := newTestNet(t)
	uri := fmt.Sprintf("tcp://127.0.0.1:%d", freePort(t))
	a := tn.member(t, func(c *Config) {
		c.Yggdrasil, c.YggdrasilPeers, c.YggdrasilListen = true, []string{}, []string{uri}
	})
	ya := a.YggdrasilAddr()
	if ya == nil {
		t.Fatal("no Yggdrasil address")
	}
	b := tn.member(t, func(c *Config) {
		c.Yggdrasil, c.YggdrasilPeers = true, []string{uri}
		c.Bootstrap = []peer.AddrInfo{{ID: a.ID(), Addrs: []ma.Multiaddr{ya}}}
	})
	eventually(t, 20*time.Second, "mutual membership over Yggdrasil", func() bool {
		return connectedMember(a, b.ID()) && connectedMember(b, a.ID())
	})
	if a.YggdrasilPeers() == 0 || b.YggdrasilPeers() == 0 {
		t.Fatal("the Yggdrasil nodes are not peered")
	}
	conns := b.host.Network().ConnsToPeer(a.ID())
	if len(conns) == 0 || !conns[0].RemoteMultiaddr().Equal(ya) {
		t.Fatalf("connections to a: %v, want one to %s", conns, ya)
	}
	got, err := echo(t.Context(), b, a.ID(), "ping")
	if err != nil || got != "ping\n" {
		t.Fatalf("echo: %q, %v", got, err)
	}
	if !containsAddr(a.AddrInfo().Addrs, ya) {
		t.Fatalf("a does not advertise %s: %v", ya, a.AddrInfo().Addrs)
	}
}

func containsAddr(addrs []ma.Multiaddr, a ma.Multiaddr) bool {
	for _, x := range addrs {
		if x.Equal(a) {
			return true
		}
	}
	return false
}

// The case Yggdrasil is for: the announcer has no address but /ygg (here:
// it listens on no IP address at all), and a newcomer that knows only the
// network finds it through the Mainline DHT and joins over Yggdrasil.
func TestMainlineAnnouncerOnlyOverYggdrasil(t *testing.T) {
	tn := newTestNet(t)
	d, err := mldht.NewTestNetwork(30)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	uri := fmt.Sprintf("tcp://127.0.0.1:%d", freePort(t))
	a := tn.member(t, withMainline(d), func(c *Config) {
		c.ListenAddrs = []string{}
		c.Yggdrasil, c.YggdrasilPeers, c.YggdrasilListen = true, []string{}, []string{uri}
	})
	if addrs := a.AddrInfo().Addrs; len(addrs) != 1 || !addrs[0].Equal(a.YggdrasilAddr()) {
		t.Fatalf("announcer addresses %v, want only its /ygg one", addrs)
	}
	item, err := SignBootstrapRecord(tn.root, tn.net, []peer.ID{a.ID()}, 1)
	if err != nil {
		t.Fatal(err)
	}
	c, err := mldht.Listen("127.0.0.1:0", d.Bootstrap())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Put(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	b := tn.member(t, withMainline(d), func(c *Config) {
		c.Yggdrasil, c.YggdrasilPeers = true, []string{uri}
	})
	eventually(t, 30*time.Second, "the newcomer to join through the announcer's /ygg address", func() bool {
		return connectedMember(b, a.ID())
	})
}
