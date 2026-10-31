package trust

import (
	"fmt"
	"testing"

	"github.com/libp2p/go-libp2p/core/peer"
)

func ids(prefix string, n int) []peer.ID {
	var out []peer.ID
	for i := range n {
		out = append(out, peer.ID(fmt.Sprintf("%s%d", prefix, i)))
	}
	return out
}

func TestEigenTrust(t *testing.T) {
	members := ids("m", 3)
	honest := ids("h", 2)
	sybils := ids("s", 20)
	rows := map[peer.ID]map[peer.ID]float64{
		// Members verified the honest guests' work; one member was also
		// fooled once into a little trust for a sybil.
		members[0]: {honest[0]: 0.7, honest[1]: 0.3},
		members[1]: {honest[0]: 0.5, honest[1]: 0.45, sybils[0]: 0.05},
		members[2]: {honest[1]: 1},
		// The honest guests verified each other.
		honest[0]: {honest[1]: 1},
		honest[1]: {honest[0]: 1},
	}
	// Sybils praise each other unanimously, and say nothing of anyone else.
	for _, s := range sybils {
		r := map[peer.ID]float64{}
		for _, o := range sybils {
			if o != s {
				r[o] = 1 / float64(len(sybils)-1)
			}
		}
		rows[s] = r
	}
	tr := eigenTrust(rows, members)
	var total, sybilTotal float64
	for p, v := range tr {
		total += v
		if p[0] == 's' {
			sybilTotal += v
		}
	}
	if total < 0.999 || total > 1.001 {
		t.Fatalf("trust sums to %f", total)
	}
	// The clique holds what leaked to it through the one fooled member,
	// circulating among itself: it cannot grow it.
	if sybilTotal > 0.05 {
		t.Fatalf("20 sybils gathered %.3f of all trust", sybilTotal)
	}
	for _, h := range honest {
		if tr[h] < 10*tr[sybils[0]] || tr[h] < 0.1 {
			t.Fatalf("honest guest %s has %.3f, a sybil %.3f", h, tr[h], tr[sybils[0]])
		}
	}
	t.Logf("member %.3f, honest %.3f %.3f, sybils %.4f in all", tr[members[0]], tr[honest[0]], tr[honest[1]], sybilTotal)

	// Without any member's support, a clique gets nothing at all.
	delete(rows[members[1]], sybils[0])
	if tr := eigenTrust(rows, members); tr[sybils[0]] != 0 {
		t.Fatalf("unsupported sybil has trust %f", tr[sybils[0]])
	}
	if tr := eigenTrust(rows, nil); len(tr) != 0 {
		t.Fatal("trust without pre-trusted peers")
	}
}

func TestSatisfaction(t *testing.T) {
	p := Peer{Verified: 30, Failed: 2}
	if p.satisfaction() != 28 {
		t.Fatal(p.satisfaction())
	}
	p.Cheated = 2
	if p.satisfaction() != 0 {
		t.Fatal("one lie should outweigh twenty honest answers")
	}
}

func TestTrustGrowsWithVolume(t *testing.T) {
	m, g := peer.ID("m"), peer.ID("g")
	s := &Service{peers: map[peer.ID]*Peer{}}
	share := func() float64 {
		return eigenTrust(map[peer.ID]map[peer.ID]float64{m: s.localTrust()}, []peer.ID{m})[g]
	}
	s.peers[g] = &Peer{ID: g, Verified: 1}
	one := share()
	s.peers[g].Verified = 50
	fifty := share()
	s.peers[g].Verified = 500
	many := share()
	if !(one < fifty/10 && fifty < many) || many < 0.4 {
		t.Fatalf("trust after 1, 50, 500 verified results: %.4f %.4f %.4f", one, fifty, many)
	}
}
