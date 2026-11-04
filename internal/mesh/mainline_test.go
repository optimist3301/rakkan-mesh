package mesh

import (
	"bytes"
	"os"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"

	"github.com/optimist3301/rakkan-mesh/internal/mldht"
)

func withMainline(d *mldht.TestNetwork) nodeOpt {
	return func(c *Config) {
		c.Mainline, c.MainlineBootstrap, c.MainlineInterval = true, d.Bootstrap(), 200*time.Millisecond
	}
}

// A node that knows nothing but the network finds an announcer through the
// DHT, from the root's record, and joins.
func TestMainlineBootstrap(t *testing.T) {
	tn := newTestNet(t)
	d, err := mldht.NewTestNetwork(30)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	a := tn.member(t, withMainline(d))
	other := tn.member(t, withBootstrap(a)) // a member, but not an announcer
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
	b := tn.member(t, withMainline(d))
	eventually(t, 20*time.Second, "the newcomer to find the announcer", func() bool { return connectedMember(b, a.ID()) })
	// Through the announcer, the rest of the network.
	eventually(t, 20*time.Second, "the newcomer to find the other member", func() bool { return connectedMember(b, other.ID()) })
}

// Records are sealed to their network: another network's key opens
// nothing, and the announcer list and addresses round-trip.
func TestMainlineRecords(t *testing.T) {
	tn, other := newTestNet(t), newTestNet(t)
	_, id := newKey(t)
	item, err := SignBootstrapRecord(tn.root, tn.net, []peer.ID{id}, 5)
	if err != nil {
		t.Fatal(err)
	}
	rec := newMainlineRecords(tn.net.ID)
	plain, err := rec.open(rec.rootSalt, item.Value)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := parseAnnouncers(plain)
	if err != nil || len(ids) != 1 || ids[0] != id {
		t.Fatalf("%v %v", ids, err)
	}
	if _, err := newMainlineRecords(other.net.ID).open(rec.rootSalt, item.Value); err == nil {
		t.Error("another network opened the record")
	}
	if _, err := rec.open(rec.addrSalt, item.Value); err == nil {
		t.Error("a root record opened as an address record")
	}
	if bytes.Equal(rec.rootSalt, newMainlineRecords(other.net.ID).rootSalt) {
		t.Error("two networks share a salt")
	}
	addrs := []ma.Multiaddr{ma.StringCast("/ip4/1.2.3.4/tcp/4001"), ma.StringCast("/ip6/::1/udp/9/quic-v1")}
	got, err := decodeAddrs(encodeAddrs(addrs))
	if err != nil || len(got) != 2 || !got[0].Equal(addrs[0]) || !got[1].Equal(addrs[1]) {
		t.Fatalf("%v %v", got, err)
	}
	if _, err := SignBootstrapRecord(tn.root, tn.net, make([]peer.ID, maxAnnouncers+1), 1); err == nil {
		t.Error("too many announcers taken")
	}
}

// The same through the real Mainline DHT (RAKKAN_MLDHT_ONLINE=1); the
// nodes are on loopback, their records in the DHT.
func TestMainlineBootstrapOnline(t *testing.T) {
	if os.Getenv("RAKKAN_MLDHT_ONLINE") == "" {
		t.Skip("RAKKAN_MLDHT_ONLINE not set")
	}
	tn := newTestNet(t)
	online := func(c *Config) { c.Mainline, c.MainlineInterval = true, 5*time.Second }
	a := tn.member(t, online)
	item, err := SignBootstrapRecord(tn.root, tn.net, []peer.ID{a.ID()}, time.Now().Unix())
	if err != nil {
		t.Fatal(err)
	}
	c, err := mldht.Listen(":0", mldht.DefaultBootstrap)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Put(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	b := tn.member(t, online)
	eventually(t, 3*time.Minute, "the newcomer to find the announcer through the DHT", func() bool { return connectedMember(b, a.ID()) })
	t.Logf("joined in %v", time.Since(start))
}
