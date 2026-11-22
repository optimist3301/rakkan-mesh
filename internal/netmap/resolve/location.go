package resolve

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// Level says how well a place is known (spec §9).
type Level int

const (
	LevelUnknown Level = iota // nothing usable
	LevelRegion               // a country/city, or a position uncertain by more than AreaMaxM
	LevelArea                 // an estimated area: uncertainty up to AreaMaxM
	LevelPrecise              // uncertainty up to PreciseMaxM
)

func (l Level) String() string {
	return [...]string{"unknown", "region", "area", "precise"}[l]
}

func (l Level) MarshalText() ([]byte, error) { return []byte(l.String()), nil }

func (l *Level) UnmarshalText(b []byte) error {
	for i, name := range [...]string{"unknown", "region", "area", "precise"} {
		if string(b) == name {
			*l = Level(i)
			return nil
		}
	}
	return fmt.Errorf("unknown location level %q", b)
}

// Tunables of the v1 rules (spec §9).
var (
	PreciseMaxM     = 30.0
	AreaMaxM        = 3000.0
	MinConf         = 0.05    // estimates below this (after decay) are expired
	MaxPropagateM   = 20000.0 // derived estimates with a larger radius are useless
	MaxHops         = 3
	conflictPenalty = 0.6
)

// hopRule says how far, and how reliably, a known position carries across
// one link of the given kind. Only explicit links carry position: sharing a
// network segment is not enough, and routed (l3) hops can be anywhere.
type hopRule struct {
	addM    float64 // added to the radius
	conf    float64 // multiplies confidence
	logical bool    // the logical place ("room 101") carries across too
}

var hopRules = map[string]hopRule{
	"l2":        {100, 0.9, true},  // cable: within a cable run
	"wifi":      {100, 0.8, false}, // radio: within AP range, which spans rooms
	"l2-behind": {300, 0.5, false}, // somewhere behind a port: a building's worth
}

// halfLife is how fast an estimate goes stale. A GPS fix is about a moment;
// a hand-entered position of a fixed device is long-lived, of the observer's
// own laptop short-lived.
func halfLife(method string, isObserver bool) time.Duration {
	day := 24 * time.Hour
	switch {
	case method == "gpsd":
		return 2 * time.Hour
	case method == "manual" && isObserver:
		return 7 * day
	case method == "manual", method == "snmp":
		return 365 * day
	}
	return 30 * day
}

type LatLon struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

// LocSource is one piece of evidence behind a Location: either a position
// someone observed (Kind "direct") or one inferred over links ("link").
type LocSource struct {
	Kind     string   `json:"kind"`
	Method   string   `json:"method,omitempty"`
	Observer string   `json:"observer,omitempty"`
	From     string   `json:"from,omitempty"` // link: the device it was carried from
	Via      []string `json:"via,omitempty"`  // link: kinds of the links crossed
	Point    LatLon   `json:"point"`
	RadiusM  float64  `json:"radius_m"`
	Conf     float64  `json:"confidence"`
	Time     int64    `json:"time,omitempty"`
}

type Location struct {
	Level       Level       `json:"level"`
	Point       *LatLon     `json:"point,omitempty"`
	RadiusM     float64     `json:"radius_m,omitempty"`
	Conf        float64     `json:"confidence"`
	Derived     bool        `json:"derived,omitempty"` // nobody observed it here; carried over links
	Logical     string      `json:"logical,omitempty"`
	LogicalFrom string      `json:"logical_from,omitempty"` // device the logical place was carried from
	Region      string      `json:"region,omitempty"`
	Conflict    bool        `json:"conflict,omitempty"` // sources that cannot all be right
	Sources     []LocSource `json:"sources,omitempty"`
}

func levelOf(radiusM float64) Level {
	switch {
	case radiusM <= PreciseMaxM:
		return LevelPrecise
	case radiusM <= AreaMaxM:
		return LevelArea
	}
	return LevelRegion
}

func distanceM(a, b LatLon) float64 {
	const earth = 6371008.8
	rad := math.Pi / 180
	dLat, dLon := (b.Lat-a.Lat)*rad, (b.Lon-a.Lon)*rad
	h := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(a.Lat*rad)*math.Cos(b.Lat*rad)*math.Sin(dLon/2)*math.Sin(dLon/2)
	return 2 * earth * math.Asin(math.Min(1, math.Sqrt(h)))
}

type fusedEst struct {
	p        LatLon
	r, conf  float64
	conflict bool
}

// fuse combines estimates of one position. Estimates that agree (their
// circles overlap) reinforce each other: the position is the average weighted
// by confidence/radius², the radius shrinks like independent measurements
// (1/√Σ1/r²), and confidence combines as 1−Π(1−c). Estimates that cannot be
// reconciled with the most precise trustworthy one are left out and mark the
// result as conflicting, which costs confidence.
func fuse(es []LocSource) fusedEst {
	maxConf := 0.0
	for _, e := range es {
		maxConf = max(maxConf, e.Conf)
	}
	var seed *LocSource
	for i := range es {
		e := &es[i]
		if e.Conf < maxConf/2 {
			continue
		}
		if seed == nil || cmp.Or(cmp.Compare(e.RadiusM, seed.RadiusM),
			cmp.Compare(e.Point.Lat, seed.Point.Lat), cmp.Compare(e.Point.Lon, seed.Point.Lon)) < 0 {
			seed = e
		}
	}
	const mPerDeg = 111320.0
	cosLat := math.Cos(seed.Point.Lat * math.Pi / 180)
	var sw, sx, sy, sInv, miss float64
	miss = 1
	n := 0
	for _, e := range es {
		if distanceM(seed.Point, e.Point) > seed.RadiusM+e.RadiusM {
			continue
		}
		n++
		w := e.Conf / (e.RadiusM * e.RadiusM)
		sw += w
		sx += w * (e.Point.Lon - seed.Point.Lon) * mPerDeg * cosLat
		sy += w * (e.Point.Lat - seed.Point.Lat) * mPerDeg
		sInv += 1 / (e.RadiusM * e.RadiusM)
		miss *= 1 - e.Conf
	}
	out := fusedEst{
		p: LatLon{
			Lat: seed.Point.Lat + sy/sw/mPerDeg,
			Lon: seed.Point.Lon + sx/sw/mPerDeg/cosLat,
		},
		r:    max(1, 1/math.Sqrt(sInv)),
		conf: 1 - miss,
	}
	if n < len(es) {
		out.conflict = true
		out.conf *= conflictPenalty
	}
	return out
}

type textClaim struct {
	v    string
	conf float64
}

func better(a, b textClaim) bool { // is a better than b
	return cmp.Or(cmp.Compare(a.conf, b.conf), -strings.Compare(a.v, b.v)) > 0
}

type edge struct{ to, kind string }

// assignLocations fills Device.Loc for every device (spec §9).
func assignLocations(g *Graph, list []*obs.Observation, devOf []string, weight func(*obs.Observation) float64) {
	byID := make(map[string]*Device, len(g.Devices))
	for _, d := range g.Devices {
		byID[d.ID] = d
	}
	var at int64 // "now" is the newest thing in the set: a pure function of the set
	for _, o := range list {
		_, t1 := span(o)
		at = max(at, t1)
	}

	direct := map[string][]LocSource{}
	logical := map[string]textClaim{}
	region := map[string]textClaim{}
	for i, o := range list {
		d := byID[devOf[i]]
		if d == nil || o.V != obs.SchemaVersion {
			continue
		}
		_, t1 := span(o)
		age := float64(max(0, at-t1))
		decay := math.Exp2(-age/float64(halfLife(o.Method, d.IsSelf).Milliseconds())) * weight(o)
		for _, c := range o.Claims {
			conf := float64(c.Conf()) / 1000 * decay
			if conf < MinConf {
				continue
			}
			switch c.K {
			case "loc.geo":
				lat, lon, r, err := obs.ParseGeo(c.V)
				if err != nil {
					continue
				}
				direct[d.ID] = append(direct[d.ID], LocSource{Kind: "direct", Method: o.Method,
					Observer: shortKey(o.Observer), Point: LatLon{lat, lon}, RadiusM: r, Conf: conf, Time: o.Time})
			case "loc.logical", "location.text":
				if t := (textClaim{sanitizeText(c.V), conf}); t.v != "" && better(t, logical[d.ID]) {
					logical[d.ID] = t
				}
			case "loc.region":
				if t := (textClaim{sanitizeText(c.V), conf}); t.v != "" && better(t, region[d.ID]) {
					region[d.ID] = t
				}
			}
		}
	}

	adj := map[string][]edge{}
	for _, l := range g.Links {
		if _, ok := hopRules[l.Kind]; ok {
			adj[l.A] = append(adj[l.A], edge{l.B, l.Kind})
			adj[l.B] = append(adj[l.B], edge{l.A, l.Kind})
		}
	}
	for _, es := range adj {
		slices.SortFunc(es, func(a, b edge) int { return cmp.Or(strings.Compare(a.to, b.to), strings.Compare(a.kind, b.kind)) })
	}

	// Carry every directly located device's position over nearby links.
	derived := map[string][]LocSource{}
	for _, id := range sortedKeys(direct) {
		src := fuse(direct[id])
		type step struct {
			node string
			hops int
			add  float64
			conf float64
			via  []string
		}
		seen := map[string]bool{id: true}
		queue := []step{{node: id, conf: src.conf}}
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			for _, e := range adj[cur.node] {
				if seen[e.to] {
					continue
				}
				rule := hopRules[e.kind]
				next := step{e.to, cur.hops + 1, cur.add + rule.addM, cur.conf * rule.conf, append(slices.Clone(cur.via), e.kind)}
				if src.r+next.add > MaxPropagateM || next.conf < MinConf {
					continue
				}
				seen[e.to] = true
				derived[e.to] = append(derived[e.to], LocSource{Kind: "link", From: id, Via: next.via,
					Point: src.p, RadiusM: src.r + next.add, Conf: next.conf})
				if next.hops < MaxHops {
					queue = append(queue, next)
				}
			}
		}
	}

	for _, d := range g.Devices {
		loc := &Location{}
		d.Loc = loc
		all := append(slices.Clone(direct[d.ID]), derived[d.ID]...)
		if len(all) > 0 {
			f := fuse(all)
			loc.Point, loc.RadiusM, loc.Conf, loc.Conflict = &f.p, f.r, f.conf, f.conflict
			loc.Level = levelOf(f.r)
			loc.Derived = len(direct[d.ID]) == 0
			slices.SortStableFunc(all, func(a, b LocSource) int {
				return cmp.Or(cmp.Compare(a.RadiusM, b.RadiusM), cmp.Compare(b.Conf, a.Conf))
			})
			loc.Sources = all[:min(len(all), 8)]
		}
		if r, ok := region[d.ID]; ok {
			loc.Region = r.v
			if loc.Point == nil {
				loc.Level, loc.Conf = LevelRegion, r.conf
			}
		}
		if t, ok := logical[d.ID]; ok {
			loc.Logical = t.v
		} else {
			// A room carries across a cable or radio link, not across "somewhere behind".
			var best textClaim
			from := ""
			for _, e := range adj[d.ID] {
				if t, ok := logical[e.to]; ok && hopRules[e.kind].logical {
					if c := (textClaim{t.v, t.conf * hopRules[e.kind].conf}); better(c, best) {
						best, from = c, e.to
					}
				}
			}
			loc.Logical, loc.LogicalFrom = best.v, from
		}
	}
}

// sanitizeText bounds free text that arrives from other people's maps.
func sanitizeText(s string) string {
	s = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, strings.ToValidUTF8(s, ""))
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > 120 {
		s = string(r[:120])
	}
	return s
}
