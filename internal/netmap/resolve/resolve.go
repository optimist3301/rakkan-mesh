// Package resolve derives the graph (devices, networks, links) from a set of
// observations (spec §6). It is a pure function of the set: the same
// observations always give the same graph, regardless of the order they were
// collected or merged in.
package resolve

import (
	"cmp"
	"encoding/hex"
	"net/netip"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

// Tunables of the v1 rules.
var (
	IPAttachWindow = 24 * time.Hour   // IP-only observation → binding distance
	CoBindWindow   = 30 * time.Minute // same IP on two clusters → same device
	MaxIPv4PerDev  = 2                // more v4 addresses ⇒ proxy-ARP/router: bindings unreliable
)

type ClaimValue struct {
	Value     string   `json:"value"`
	Conf      uint16   `json:"conf"`
	First     int64    `json:"first"`
	Last      int64    `json:"last"`
	Count     int      `json:"count"`
	Observers []string `json:"observers"`
	Modes     []string `json:"modes"`
}

type Addr struct {
	IP    string `json:"ip"`
	Net   string `json:"net"`
	First int64  `json:"first"`
	Last  int64  `json:"last"`
}

type Evidence struct {
	A, B   string `json:"-"`
	Rule   string `json:"rule"`
	Detail string `json:"detail"`
}

type Device struct {
	ID        string                   `json:"id"`
	Label     string                   `json:"label"`
	Idents    []string                 `json:"idents"`
	Addrs     []Addr                   `json:"addrs,omitempty"`
	Networks  []string                 `json:"networks"`
	Claims    map[string][]*ClaimValue `json:"claims"`
	First     int64                    `json:"first"`
	Last      int64                    `json:"last"`
	Obs       int                      `json:"observations"`
	Observers []string                 `json:"observers"`
	Modes     []string                 `json:"modes"`
	Methods   []string                 `json:"methods"`
	Evidence  []Evidence               `json:"evidence,omitempty"`
	IsSelf    bool                     `json:"is_observer,omitempty"`
	Loc       *Location                `json:"location"`
	AS        *AS                      `json:"as,omitempty"`
	Class     *Class                   `json:"class,omitempty"`
}

// Best returns the best value of a claim: highest confidence, then latest.
func (d *Device) Best(k string) string {
	vs := d.Claims[k]
	if len(vs) == 0 {
		return ""
	}
	return vs[0].Value
}

type Network struct {
	ID        string                   `json:"id"`
	Scoped    bool                     `json:"observer_local,omitempty"`
	Claims    map[string][]*ClaimValue `json:"claims"`
	Devices   []string                 `json:"devices"`
	Observers []string                 `json:"observers"`
	AS        *AS                      `json:"as,omitempty"` // of its external address
}

func (n *Network) Best(k string) string {
	if vs := n.Claims[k]; len(vs) > 0 {
		return vs[0].Value
	}
	return ""
}

type Link struct {
	A      string `json:"a"`
	B      string `json:"b"`
	Kind   string `json:"kind"`
	Detail string `json:"detail,omitempty"`
}

type Conflict struct {
	Rule   string `json:"rule"`
	A      string `json:"a"`
	B      string `json:"b"`
	Reason string `json:"reason"`
}

type Graph struct {
	Networks     []*Network     `json:"networks"`
	Devices      []*Device      `json:"devices"`
	Links        []*Link        `json:"links"`
	Conflicts    []Conflict     `json:"conflicts,omitempty"`
	Observations int            `json:"observations"`
	Observers    []string       `json:"observers"`
	ObserverInfo []ObserverInfo `json:"observer_info"`
	ASes         []*ASGroup     `json:"ases,omitempty"`

	checks map[string]*Check // reputation of the observers that were checked
}

func (g *Graph) Device(id string) *Device {
	for _, d := range g.Devices {
		if d.ID == id || strings.HasPrefix(d.ID, id) {
			return d
		}
	}
	return nil
}

// ---- union-find with per-cluster strong identifiers for conflict checks ----

type uf struct {
	parent map[string]string
	strong map[string]map[string][]string // root → kind → values
}

func newUF() *uf { return &uf{parent: map[string]string{}, strong: map[string]map[string][]string{}} }

func (u *uf) add(id string) {
	if _, ok := u.parent[id]; ok {
		return
	}
	u.parent[id] = id
	if obs.StrengthOf(id) == obs.Strong {
		u.strong[id] = map[string][]string{obs.Kind(id): {id}}
	}
}

func (u *uf) find(id string) string {
	p, ok := u.parent[id]
	if !ok {
		return ""
	}
	if p == id {
		return id
	}
	r := u.find(p)
	u.parent[id] = r
	return r
}

// conflict reports a strong kind holding different values in both clusters.
func (u *uf) conflict(a, b string, kinds map[string]bool) string {
	sa, sb := u.strong[a], u.strong[b]
	for k, va := range sa {
		if !kinds[k] {
			continue
		}
		if vb, ok := sb[k]; ok && !slices.Equal(va, vb) {
			return k + ": " + va[0] + " ≠ " + vb[0]
		}
	}
	return ""
}

func (u *uf) union(a, b string) bool {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return false
	}
	if ra > rb { // deterministic root choice
		ra, rb = rb, ra
	}
	u.parent[rb] = ra
	if u.strong[ra] == nil {
		u.strong[ra] = map[string][]string{}
	}
	for k, vs := range u.strong[rb] {
		merged := append(u.strong[ra][k], vs...)
		slices.Sort(merged)
		u.strong[ra][k] = slices.Compact(merged)
	}
	delete(u.strong, rb)
	return true
}

// Identity kinds a device has exactly one of. Medium-strength merges must not
// join clusters holding different values of these.
var exclusiveKinds = map[string]bool{"rkm": true, "ssh": true, "lldp": true}

// For IP co-binding, MACs count too: one IP bound to two different NICs at
// the same moment is more likely DHCP churn than one device. UPnP UUIDs do
// not: one device announces a root UUID plus one per embedded device.
var cobindKinds = map[string]bool{"rkm": true, "ssh": true, "lldp": true, "mac": true, "lmac": true}

var l2Methods = map[string]bool{"neigh": true, "arp": true, "dhcp": true, "ndp": true, "lldp": true, "wifi-assoc": true}

type binding struct {
	ip     string
	anchor string
	t0, t1 int64
	obsIdx int
}

func span(o *obs.Observation) (int64, int64) {
	if o.Until > 0 {
		return o.Time, o.Until
	}
	return o.Time, o.Time
}

func shortKey(b []byte) string { return hex.EncodeToString(b)[:12] }

// Build resolves a set of observations into a graph, trusting every signer.
func Build(list []*obs.Observation) *Graph { return BuildWith(list, Options{}) }

// BuildWith resolves a set as seen by a viewer with the given trust.
//
// Observers that are not trusted (unknown ones, and above all guests of the
// network) are checked against those that are: where they state a fact the
// trusted observers contradict, their weight falls, and past a point they
// are left out. See crossCheck.
func BuildWith(list []*obs.Observation, opts Options) *Graph {
	g := build(list, opts, nil)
	if len(g.checks) == 0 {
		return g
	}
	// The weights depend on the first result, and the second result on the
	// weights: one round is enough, because the check compares only with
	// observers nobody discounts.
	return build(list, opts, g.checks)
}

func build(list []*obs.Observation, opts Options, rep map[string]*Check) *Graph {
	list = slices.Clone(list)
	slices.SortFunc(list, func(a, b *obs.Observation) int {
		return cmp.Or(cmp.Compare(a.Time, b.Time), strings.Compare(string(a.ID[:]), string(b.ID[:])))
	})
	g := &Graph{Observations: len(list)}
	list, tc, infos := prepare(list, opts, rep)
	g.ObserverInfo = infos
	disputes := map[string]map[string]bool{} // observer → identity merges the trusted contradict
	u := newUF()
	evidence := []Evidence{}
	observers := map[string]bool{}

	// Classify subject identifiers per observation.
	type parts struct{ nets, ips, strong, medium []string }
	ps := make([]parts, len(list))
	for i, o := range list {
		if o.V != obs.SchemaVersion {
			continue
		}
		observers[shortKey(o.Observer)] = true
		for _, id := range o.Subject {
			switch k := obs.Kind(id); {
			case k == "net":
				ps[i].nets = append(ps[i].nets, id)
			case obs.IsIPLike(id):
				ps[i].ips = append(ps[i].ips, id)
			case obs.StrengthOf(id) == obs.Strong:
				ps[i].strong = append(ps[i].strong, id)
				u.add(id)
			default:
				ps[i].medium = append(ps[i].medium, id)
				u.add(id)
			}
		}
	}

	// Pass 1: strong identifiers asserted together are one entity. Trusted
	// signers first; then unknown ones, who may not merge what trusted
	// evidence keeps apart (else anyone could glue a victim's MAC to their
	// own SSH key). The two sweeps keep the result independent of order.
	for _, trustedSweep := range []bool{true, false} {
		for i, p := range ps {
			if (tc.of(list[i]) >= trust.Trusted) != trustedSweep {
				continue
			}
			for _, id := range p.strong[min(1, len(p.strong)):] {
				ra, rb := u.find(p.strong[0]), u.find(id)
				if ra == rb {
					continue
				}
				if !trustedSweep {
					if why := u.conflict(ra, rb, cobindKinds); why != "" {
						g.Conflicts = append(g.Conflicts, Conflict{"untrusted-identity", p.strong[0], id,
							why + " (asserted by unknown observer " + shortKey(list[i].Observer) + ")"})
						k := hex.EncodeToString(list[i].Observer)
						if disputes[k] == nil {
							disputes[k] = map[string]bool{}
						}
						disputes[k][p.strong[0]+" "+id] = true
						continue
					}
				}
				u.union(ra, rb)
				evidence = append(evidence, Evidence{p.strong[0], id, "same-observation", list[i].Method})
			}
		}
	}
	// Pass 2: medium identifiers join unless that contradicts strong ones.
	for i, p := range ps {
		ids := append(slices.Clone(p.strong), p.medium...)
		if len(p.medium) == 0 || len(ids) < 2 {
			continue
		}
		for _, id := range ids[1:] {
			ra, rb := u.find(ids[0]), u.find(id)
			if ra == rb {
				continue
			}
			if why := u.conflict(ra, rb, exclusiveKinds); why != "" {
				g.Conflicts = append(g.Conflicts, Conflict{"medium-identifier", ids[0], id, why})
				continue
			}
			u.union(ra, rb)
			evidence = append(evidence, Evidence{ids[0], id, "medium-identifier", list[i].Method})
		}
	}

	// IP bindings: (ip, cluster, time) from observations that name both.
	var binds []binding
	for i, p := range ps {
		anchor := firstOf(p.strong, p.medium)
		if anchor == "" {
			continue
		}
		t0, t1 := span(list[i])
		for _, ip := range p.ips {
			binds = append(binds, binding{ip, anchor, t0, t1, i})
		}
	}
	// Clusters bound to many IPv4 addresses (routers doing proxy ARP, VRRP
	// MACs) do not reliably identify who holds an IP.
	unreliable := func() map[string]bool {
		v4 := map[string]map[string]bool{}
		for _, b := range binds {
			r := u.find(b.anchor)
			if a, err := netip.ParseAddr(obs.Value(b.ip)); err == nil && a.Is4() {
				if v4[r] == nil {
					v4[r] = map[string]bool{}
				}
				v4[r][b.ip] = true
			}
		}
		out := map[string]bool{}
		for r, ips := range v4 {
			if len(ips) > MaxIPv4PerDev {
				out[r] = true
			}
		}
		return out
	}()

	// Pass 3: the same IP bound to two clusters at the same time → one device.
	byIP := map[string][]binding{}
	for _, b := range binds {
		if !unreliable[u.find(b.anchor)] {
			byIP[b.ip] = append(byIP[b.ip], b)
		}
	}
	w := CoBindWindow.Milliseconds()
	for _, ip := range sortedKeys(byIP) {
		bs := byIP[ip]
		for i := 0; i < len(bs); i++ {
			for j := i + 1; j < len(bs); j++ {
				a, b := bs[i], bs[j]
				if a.t0-w > b.t1 || b.t0-w > a.t1 {
					continue
				}
				ra, rb := u.find(a.anchor), u.find(b.anchor)
				if ra == rb {
					continue
				}
				if why := u.conflict(ra, rb, cobindKinds); why != "" {
					g.Conflicts = append(g.Conflicts, Conflict{"ip-cobinding", a.anchor, b.anchor, obs.Value(ip) + ": " + why})
					continue
				}
				u.union(ra, rb)
				evidence = append(evidence, Evidence{a.anchor, b.anchor, "ip-cobinding",
					obs.Value(ip) + " (" + list[a.obsIdx].Method + "+" + list[b.obsIdx].Method + ")"})
			}
		}
	}

	// Assign each observation to a cluster root.
	byIPAll := map[string][]binding{}
	for _, b := range binds {
		byIPAll[b.ip] = append(byIPAll[b.ip], b)
	}
	// boundTo finds the cluster an address belonged to around time t (the
	// binding nearest in time within IPAttachWindow), or "".
	boundTo := func(ip string, t int64) string {
		best, bestD := "", IPAttachWindow.Milliseconds()+1
		for _, b := range byIPAll[ip] {
			r := u.find(b.anchor)
			if unreliable[r] && hasReliable(byIPAll[ip], u, unreliable) {
				continue
			}
			d := max(b.t0-t, t-b.t1, 0)
			if d < bestD || (d == bestD && r < best) {
				best, bestD = r, d
			}
		}
		return best
	}
	owner := make([]string, len(list))
	for i, p := range ps {
		if a := firstOf(p.strong, p.medium); a != "" {
			owner[i] = u.find(a)
			continue
		}
		if len(p.ips) == 0 {
			continue
		}
		ip := p.ips[0]
		t0, _ := span(list[i])
		best := boundTo(ip, t0)
		if best == "" {
			u.add(ip) // IP-only device
			for _, other := range p.ips[1:] {
				u.add(other)
				u.union(ip, other)
			}
			best = u.find(ip)
		}
		owner[i] = best
	}

	if rep == nil {
		g.checks = crossCheck(list, owner, tc, disputes)
	}

	// Build devices.
	devs := map[string]*Device{}
	nets := map[string]*Network{}
	getNet := func(id string) *Network {
		n := nets[id]
		if n == nil {
			n = &Network{ID: id, Scoped: strings.HasPrefix(id, "x"), Claims: map[string][]*ClaimValue{}}
			nets[id] = n
		}
		return n
	}
	devNets := map[string]map[string]bool{}
	devObservers := map[string]map[string]bool{}
	devModes := map[string]map[string]bool{}
	devMethods := map[string]map[string]bool{}
	addrs := map[string]map[string]*Addr{}

	for i, o := range list {
		if o.V != obs.SchemaVersion {
			continue
		}
		observer := shortKey(o.Observer)
		t0, t1 := span(o)
		for _, nid := range ps[i].nets {
			n := getNet(obs.Scope(nid))
			mergeClaims(n.Claims, o, observer, tc.weight(o))
			n.Observers = appendUniq(n.Observers, observer)
		}
		r := owner[i]
		if r == "" {
			continue
		}
		d := devs[r]
		if d == nil {
			d = &Device{Claims: map[string][]*ClaimValue{}, First: t0, Last: t1}
			devs[r] = d
			devNets[r], devObservers[r], devModes[r], devMethods[r] = map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
			addrs[r] = map[string]*Addr{}
		}
		d.Obs++
		d.First, d.Last = min(d.First, t0), max(d.Last, t1)
		devObservers[r][observer] = true
		devModes[r][o.Mode.String()] = true
		devMethods[r][o.Method] = true
		mergeClaims(d.Claims, o, observer, tc.weight(o))
		if o.Vantage != nil && o.Vantage.Net != "" && l2Methods[o.Method] {
			devNets[r][o.Vantage.Net] = true
		}
		for _, id := range o.Subject {
			// Membership comes from on-link identifiers only: a routed
			// address (rip) is seen *from* the network, not *in* it.
			if s := obs.Scope(id); s != "" && obs.Kind(id) != "net" && obs.Kind(id) != "rip" {
				devNets[r][s] = true
			}
		}
		// Addresses: only from observations that bind them to this device.
		for _, ip := range ps[i].ips {
			if unreliable[r] && o.Method != "route" && len(ps[i].strong)+len(ps[i].medium) > 0 {
				continue // a proxy-ARP router does not "own" every IP it answers for
			}
			a := addrs[r][ip]
			if a == nil {
				a = &Addr{IP: obs.Value(ip), Net: obs.Scope(ip), First: t0, Last: t1}
				addrs[r][ip] = a
			}
			a.First, a.Last = min(a.First, t0), max(a.Last, t1)
		}
		if ps[i].strong != nil && slices.ContainsFunc(ps[i].strong, func(s string) bool { return obs.Kind(s) == "rkm" }) {
			d.IsSelf = true
		}
	}
	if len(devs) == 0 && len(nets) == 0 {
		return g
	}

	members := map[string][]string{}
	for id := range u.parent {
		if r := u.find(id); devs[r] != nil {
			members[r] = append(members[r], id)
		}
	}
	rootToDev := map[string]string{}
	for r, d := range devs {
		ids := members[r]
		slices.SortFunc(ids, func(a, b string) int {
			return cmp.Or(cmp.Compare(obs.StrengthOf(b), obs.StrengthOf(a)), strings.Compare(a, b))
		})
		d.Idents = ids
		if len(ids) > 0 {
			d.ID = obs.HashID("d", ids[0])
		}
		d.Networks = sortedKeys(devNets[r])
		d.Observers = sortedKeys(devObservers[r])
		d.Modes = sortedKeys(devModes[r])
		d.Methods = sortedKeys(devMethods[r])
		for _, a := range addrs[r] {
			d.Addrs = append(d.Addrs, *a)
		}
		slices.SortFunc(d.Addrs, func(a, b Addr) int { return cmp.Compare(b.Last, a.Last) })
		for _, vs := range d.Claims {
			sortValues(vs)
		}
		d.Label = label(d)
		rootToDev[r] = d.ID
		for _, nid := range d.Networks {
			n := getNet(nid)
			n.Devices = append(n.Devices, d.ID)
		}
		g.Devices = append(g.Devices, d)
	}
	for _, e := range evidence {
		if r := u.find(e.A); devs[r] != nil {
			devs[r].Evidence = append(devs[r].Evidence, e)
		}
	}

	// Links from link.* claims.
	seenLink := map[string]bool{}
	for i, o := range list {
		from := rootToDev[owner[i]]
		if from == "" {
			continue
		}
		for _, c := range o.Claims {
			if !strings.HasPrefix(c.K, "link.") {
				continue
			}
			target := u.find(c.V)
			if target == "" && obs.IsIPLike(c.V) {
				// An address bound to a device (a gateway's IP as a first hop).
				target = boundTo(c.V, o.Time)
			}
			to := rootToDev[target]
			if to == "" || to == from {
				continue
			}
			detail := linkDetail(o)
			key := from + to + c.K + detail
			if !seenLink[key] {
				seenLink[key] = true
				g.Links = append(g.Links, &Link{A: from, B: to, Kind: strings.TrimPrefix(c.K, "link."), Detail: detail})
			}
		}
	}

	devOf := make([]string, len(list))
	for i := range list {
		devOf[i] = rootToDev[owner[i]]
	}
	assignLocations(g, list, devOf, tc.weight)

	for _, n := range nets {
		for _, vs := range n.Claims {
			sortValues(vs)
		}
		slices.Sort(n.Devices)
		g.Networks = append(g.Networks, n)
	}
	slices.SortFunc(g.Networks, func(a, b *Network) int { return strings.Compare(a.ID, b.ID) })
	slices.SortFunc(g.Devices, func(a, b *Device) int {
		return cmp.Or(compareIP(a, b), strings.Compare(a.ID, b.ID))
	})
	g.Observers = sortedKeys(observers)
	g.Conflicts = slices.CompactFunc(g.Conflicts, func(a, b Conflict) bool { return a == b })
	classifyAll(g)
	if opts.ASN != nil {
		annotateAS(g, opts.ASN)
	}
	return g
}

// linkDetail renders port names: link.port.a is on the observed entity's
// side, link.port.b on the side of the identifier the link points to.
func linkDetail(o *obs.Observation) string {
	a, _ := o.Get("link.port.a")
	b, _ := o.Get("link.port.b")
	switch {
	case a != "" && b != "":
		return a + " ↔ " + b
	case a != "":
		return a + " ↔ ?"
	case b != "":
		return "? ↔ " + b
	}
	if ttl, ok := o.Get("trace.ttl"); ok {
		return "ttl " + ttl
	}
	return ""
}

func hasReliable(bs []binding, u *uf, unreliable map[string]bool) bool {
	for _, b := range bs {
		if !unreliable[u.find(b.anchor)] {
			return true
		}
	}
	return false
}

func firstOf(lists ...[]string) string {
	for _, l := range lists {
		if len(l) > 0 {
			return l[0]
		}
	}
	return ""
}

func mergeClaims(m map[string][]*ClaimValue, o *obs.Observation, observer string, weight float64) {
	t0, t1 := span(o)
	for _, c := range o.Claims {
		var cv *ClaimValue
		for _, v := range m[c.K] {
			if v.Value == c.V {
				cv = v
				break
			}
		}
		if cv == nil {
			cv = &ClaimValue{Value: c.V, First: t0, Last: t1}
			m[c.K] = append(m[c.K], cv)
		}
		cv.Conf = max(cv.Conf, uint16(float64(c.Conf())*weight+0.5))
		cv.First, cv.Last = min(cv.First, t0), max(cv.Last, t1)
		cv.Count++
		cv.Observers = appendUniq(cv.Observers, observer)
		cv.Modes = appendUniq(cv.Modes, o.Mode.String())
	}
}

func sortValues(vs []*ClaimValue) {
	sort.SliceStable(vs, func(i, j int) bool {
		if vs[i].Conf != vs[j].Conf {
			return vs[i].Conf > vs[j].Conf
		}
		if vs[i].Last != vs[j].Last {
			return vs[i].Last > vs[j].Last
		}
		return vs[i].Value < vs[j].Value
	})
}

func label(d *Device) string {
	for _, k := range []string{"name", "hostname", "mdns.name", "mdns.instance", "ssid", "rdns", "model"} {
		if v := d.Best(k); v != "" {
			return v
		}
	}
	for _, a := range d.Addrs {
		if !strings.Contains(a.IP, ":") {
			return a.IP
		}
	}
	if len(d.Addrs) > 0 {
		return d.Addrs[0].IP
	}
	if len(d.Idents) > 0 {
		if obs.Kind(d.Idents[0]) == "rkm" {
			return "observer " + obs.Value(d.Idents[0])[:min(8, len(obs.Value(d.Idents[0])))]
		}
		return obs.Value(d.Idents[0])
	}
	return d.ID
}

func compareIP(a, b *Device) int {
	ip := func(d *Device) netip.Addr {
		for _, x := range d.Addrs {
			if p, err := netip.ParseAddr(x.IP); err == nil && p.Is4() {
				return p
			}
		}
		return netip.Addr{}
	}
	pa, pb := ip(a), ip(b)
	switch {
	case pa.IsValid() && !pb.IsValid():
		return -1
	case !pa.IsValid() && pb.IsValid():
		return 1
	}
	return pa.Compare(pb)
}

func appendUniq(s []string, v string) []string {
	if slices.Contains(s, v) {
		return s
	}
	return append(s, v)
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}
