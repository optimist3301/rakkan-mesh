package resolve

import (
	"net/netip"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// fake reference data: three ISPs and a transit network
func fakeASN(a netip.Addr) (AS, bool) {
	switch {
	case netip.MustParsePrefix("81.2.0.0/16").Contains(a):
		return AS{20712, "ANDREWS-ARNOLD", "GB"}, true
	case netip.MustParsePrefix("89.24.0.0/16").Contains(a):
		return AS{16019, "VODAFONE-CZ", "CZ"}, true
	case netip.MustParsePrefix("62.115.0.0/16").Contains(a):
		return AS{1299, "TWELVE99", "SE"}, true
	case netip.MustParsePrefix("9.9.9.0/24").Contains(a):
		return AS{19281, "QUAD9", "US"}, true
	}
	return AS{}, false
}

// Two observers in different cities and ISPs trace to the same target.
// Their maps meet on the shared transit router; addresses get their AS.
func TestMapsMeetOnInternetLayer(t *testing.T) {
	prague, london := key(1), key(2)
	b := &builder{t: t}
	addHop := func(k byte, net, id, prev string, ttl int) {
		claims := []obs.Claim{{K: "link.l3", V: prev}, {K: "trace.ttl", V: string(rune('0' + ttl))}}
		o, err := obs.Sign(obs.Body{Time: t0, Mode: obs.ModeActive, Method: "trace", Vantage: &obs.Vantage{Net: net},
			Subject: []string{id}, Claims: claims}, key(k))
		if err != nil {
			t.Fatal(err)
		}
		b.list = append(b.list, o)
	}
	// each observer's own LAN and gateway
	b.add(prague, obs.ModeLocal, "self", 0, []string{"rkm:aa", "mac:00:11:32:00:00:01", "ip:" + net1 + "/192.168.1.10"})
	b.add(prague, obs.ModeLocal, "route", 0, []string{"lmac:" + net1 + "/02:00:00:00:00:01", "ip:" + net1 + "/192.168.1.1"},
		obs.Claim{K: "role", V: "gateway"})
	b.add(prague, obs.ModeLocal, "route", 0, []string{"net:" + net1}, obs.Claim{K: "prefix", V: "192.168.1.0/24"})
	b.add(prague, obs.ModeActive, "wan", 0, []string{"net:" + net1}, obs.Claim{K: "wan.nat-pmp", V: "89.24.10.20"})
	b.add(london, obs.ModeLocal, "self", 0, []string{"rkm:bb", "mac:00:11:32:00:00:02", "ip:nlon/10.0.0.10"})
	b.add(london, obs.ModeLocal, "route", 0, []string{"net:nlon"}, obs.Claim{K: "prefix", V: "10.0.0.0/24"})
	b.add(london, obs.ModeActive, "wan", 0, []string{"net:nlon"}, obs.Claim{K: "wan.upnp", V: "81.2.69.160"})

	addHop(1, net1, "ip:"+net1+"/192.168.1.1", "rkm:aa", 1)
	addHop(1, net1, "gip:89.24.0.1", "ip:"+net1+"/192.168.1.1", 2)
	addHop(1, net1, "gip:62.115.40.1", "gip:89.24.0.1", 3) // shared transit router
	addHop(1, net1, "gip:9.9.9.9", "gip:62.115.40.1", 4)
	addHop(2, "nlon", "gip:81.2.0.1", "rkm:bb", 1)
	addHop(2, "nlon", "gip:62.115.40.1", "gip:81.2.0.1", 2)
	addHop(2, "nlon", "gip:9.9.9.9", "gip:62.115.40.1", 3)
	_ = time.Second

	g := BuildWith(b.list, Options{ASN: fakeASN})

	transit := byIdent(t, g, "gip:62.115.40.1")
	if transit.AS == nil || transit.AS.Number != 1299 || len(transit.Observers) != 2 {
		t.Fatalf("shared transit router: %+v observers %v", transit.AS, transit.Observers)
	}
	want := map[int]int{16019: 1, 20712: 1, 1299: 1, 19281: 1} // devices per AS
	for _, gr := range g.ASes {
		if len(gr.Devices) != want[gr.Number] {
			t.Errorf("AS%d %s: %d devices", gr.Number, gr.Name, len(gr.Devices))
		}
		delete(want, gr.Number)
	}
	if len(want) != 0 {
		t.Fatalf("missing AS groups: %v", want)
	}
	for _, n := range g.Networks {
		if n.AS == nil {
			t.Fatalf("network %s has no AS for its external address", n.ID)
		}
	}

	// The Prague observer and the London observer are connected in the graph.
	adj := map[string][]string{}
	for _, l := range g.Links {
		adj[l.A] = append(adj[l.A], l.B)
		adj[l.B] = append(adj[l.B], l.A)
	}
	from, to := byIdent(t, g, "rkm:aa").ID, byIdent(t, g, "rkm:bb").ID
	seen := map[string]bool{from: true}
	queue := []string{from}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, n := range adj[cur] {
			if !seen[n] {
				seen[n] = true
				queue = append(queue, n)
			}
		}
	}
	if !seen[to] {
		t.Fatal("maps from two cities did not join through the shared router")
	}
	// Without reference data nothing is annotated, and nothing breaks.
	if g := Build(b.list); len(g.ASes) != 0 || byIdent(t, g, "gip:62.115.40.1").AS != nil {
		t.Fatal("AS appeared without a dataset")
	}
}
