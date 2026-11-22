package resolve

import (
	"crypto/ed25519"
	"fmt"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// synth builds a realistic mix: nets × hosts, each host seen by ARP (MAC↔IP),
// ping, mDNS name and TCP, several times over a day, by `observers` keys.
func synth(tb testing.TB, nets, hosts, observers, rounds int) []*obs.Observation {
	tb.Helper()
	keys := make([]ed25519.PrivateKey, observers)
	for i := range keys {
		keys[i] = key(byte(10 + i))
	}
	var list []*obs.Observation
	sign := func(k ed25519.PrivateKey, b obs.Body) {
		o, err := obs.Sign(b, k)
		if err != nil {
			tb.Fatal(err)
		}
		list = append(list, o)
	}
	for n := range nets {
		netid := fmt.Sprintf("nsynth%05d", n)
		for h := range hosts {
			mac := fmt.Sprintf("mac:00:11:32:%02x:%02x:%02x", n>>8, n&0xff, h)
			ip := fmt.Sprintf("ip:%s/10.%d.%d.%d", netid, n>>8, n&0xff, h+1)
			name := fmt.Sprintf("mdns:%s/host-%d-%d.local", netid, n, h)
			for r := range rounds {
				k := keys[(n+r)%observers]
				t := t0 + int64(r)*int64(time.Hour/time.Millisecond)
				v := &obs.Vantage{Net: netid}
				sign(k, obs.Body{Time: t, Mode: obs.ModeActive, Method: "neigh", Vantage: v, Subject: []string{mac, ip}})
				sign(k, obs.Body{Time: t + 1, Mode: obs.ModeActive, Method: "ping", Vantage: v, Subject: []string{ip},
					Claims: []obs.Claim{{K: "ttl", V: "64"}}})
				sign(k, obs.Body{Time: t + 2, Mode: obs.ModePassive, Method: "mdns", Vantage: v, Subject: []string{name, ip},
					Claims: []obs.Claim{{K: "mdns.name", V: name}}})
				sign(k, obs.Body{Time: t + 3, Mode: obs.ModeActive, Method: "tcp", Vantage: v, Subject: []string{ip},
					Claims: []obs.Claim{{K: "port", V: "tcp/22"}}})
			}
		}
	}
	return list
}

// TestScaleSmoke keeps an eye on the order of growth in ordinary test runs.
func TestScaleSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	list := synth(t, 20, 50, 3, 5) // 20 000 observations, 1 000 devices
	start := time.Now()
	g := Build(list)
	if len(g.Devices) != 1000 {
		t.Fatalf("got %d devices", len(g.Devices))
	}
	t.Logf("%d observations → %d devices in %v", len(list), len(g.Devices), time.Since(start))
}

func BenchmarkBuild(b *testing.B) {
	for _, n := range []int{20, 100, 400} {
		list := synth(b, n, 50, 3, 5)
		b.Run(fmt.Sprintf("obs=%d", len(list)), func(b *testing.B) {
			for b.Loop() {
				Build(list)
			}
		})
	}
}
