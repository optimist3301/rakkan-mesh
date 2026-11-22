package resolve

import (
	"cmp"
	"net/netip"
	"slices"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// AS is an autonomous system. It comes from the viewer's reference data
// (Options.ASN), never from observations: it is how the internet routes an
// address today, not something anyone saw.
type AS struct {
	Number  int    `json:"asn"`
	Name    string `json:"name"`
	Country string `json:"country,omitempty"` // registration, not location
}

// ASGroup is everything in the graph that sits in one AS.
type ASGroup struct {
	AS
	Devices  []string `json:"devices"`
	Networks []string `json:"networks"`
}

// publicAddrs lists a device's globally routable addresses.
func publicAddrs(d *Device) []netip.Addr {
	var out []netip.Addr
	for _, id := range d.Idents {
		if obs.Kind(id) == "gip" {
			if a, err := netip.ParseAddr(obs.Value(id)); err == nil {
				out = append(out, a)
			}
		}
	}
	for _, a := range d.Addrs {
		if p, err := netip.ParseAddr(a.IP); err == nil && obs.IsPublic(p) && !slices.Contains(out, p) {
			out = append(out, p)
		}
	}
	return out
}

// wanAddrs: what a network's gateway reported as its outside, best first
// (as seen from the internet, then what the router believes).
func wanAddrs(n *Network) []netip.Addr {
	var out []netip.Addr
	for _, k := range []string{"wan.stun", "wan.nat-pmp", "wan.upnp"} {
		for _, v := range n.Claims[k] {
			if a, err := netip.ParseAddr(v.Value); err == nil && !slices.Contains(out, a) {
				out = append(out, a)
			}
		}
	}
	return out
}

func annotateAS(g *Graph, look func(netip.Addr) (AS, bool)) {
	groups := map[int]*ASGroup{}
	group := func(as AS) *ASGroup {
		gr := groups[as.Number]
		if gr == nil {
			gr = &ASGroup{AS: as, Devices: []string{}, Networks: []string{}} // never null in JSON
			groups[as.Number] = gr
		}
		return gr
	}
	for _, d := range g.Devices {
		for _, a := range publicAddrs(d) {
			if as, ok := look(a); ok {
				d.AS = &as
				gr := group(as)
				gr.Devices = append(gr.Devices, d.ID)
				break
			}
		}
	}
	for _, n := range g.Networks {
		for _, a := range wanAddrs(n) {
			if !obs.IsPublic(a) {
				continue
			}
			if as, ok := look(a); ok {
				n.AS = &as
				gr := group(as)
				gr.Networks = append(gr.Networks, n.ID)
				break
			}
		}
	}
	for _, gr := range groups {
		slices.Sort(gr.Devices)
		slices.Sort(gr.Networks)
		g.ASes = append(g.ASes, gr)
	}
	slices.SortFunc(g.ASes, func(a, b *ASGroup) int { return cmp.Compare(a.Number, b.Number) })
}
