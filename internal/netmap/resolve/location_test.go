package resolve

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

func geo(lat, lon, r float64) obs.Claim {
	return obs.Claim{K: "loc.geo", V: obs.FormatGeo(lat, lon, r)}
}

func byIdent(t *testing.T, g *Graph, ident string) *Device {
	t.Helper()
	for _, d := range g.Devices {
		for _, id := range d.Idents {
			if id == ident {
				return d
			}
		}
	}
	t.Fatalf("no device with %s", ident)
	return nil
}

const (
	swMAC   = "mac:00:1b:54:00:00:01"
	hostMAC = "mac:00:11:32:aa:aa:01"
	farMAC  = "mac:00:11:32:aa:aa:02"
	rtrMAC  = "mac:00:11:32:aa:aa:03"
)

// A located switch with three neighbours over three kinds of link.
func switchWithNeighbours(b *builder) {
	A := key(1)
	b.add(A, obs.ModeLocal, "manual", 0, []string{swMAC}, geo(50.0870, 14.4200, 10))
	b.add(A, obs.ModeActive, "snmp", 0, []string{hostMAC}, obs.Claim{K: "link.l2", V: swMAC})
	b.add(A, obs.ModeActive, "snmp", 0, []string{farMAC}, obs.Claim{K: "link.l2-behind", V: swMAC})
	b.add(A, obs.ModeActive, "trace", 0, []string{rtrMAC}, obs.Claim{K: "link.l3", V: swMAC})
}

func TestLocationLevels(t *testing.T) {
	b := &builder{t: t}
	switchWithNeighbours(b)
	g := Build(b.list)

	sw := byIdent(t, g, swMAC).Loc
	if sw.Level != LevelPrecise || sw.Derived || math.Abs(sw.RadiusM-10) > 0.01 || sw.Conf < 0.99 {
		t.Fatalf("switch: %+v", sw)
	}
	// Over a cable the position carries with 100 m of extra doubt: an area.
	host := byIdent(t, g, hostMAC).Loc
	if host.Level != LevelArea || !host.Derived || math.Abs(host.RadiusM-110) > 0.01 || math.Abs(host.Conf-0.9) > 0.01 {
		t.Fatalf("host over l2: %+v", host)
	}
	if *host.Point != *sw.Point || host.Sources[0].Kind != "link" || host.Sources[0].Via[0] != "l2" {
		t.Fatalf("host sources: %+v", host.Sources)
	}
	// "Somewhere behind a port" is a weaker, wider claim.
	far := byIdent(t, g, farMAC).Loc
	if far.Level != LevelArea || math.Abs(far.RadiusM-310) > 0.01 || math.Abs(far.Conf-0.5) > 0.01 {
		t.Fatalf("host behind: %+v", far)
	}
	// A routed hop can be anywhere: nothing carries over.
	if rtr := byIdent(t, g, rtrMAC).Loc; rtr.Level != LevelUnknown || rtr.Point != nil {
		t.Fatalf("l3 neighbour must stay unknown: %+v", rtr)
	}
}

func TestLocationFusionAgreeAndConflict(t *testing.T) {
	b := &builder{t: t}
	// Two independent observers place the same AP about 8 m apart.
	b.add(key(1), obs.ModeLocal, "manual", 0, []string{"bssid:00:aa:bb:00:00:01"}, geo(50.087000, 14.420000, 20))
	b.add(key(2), obs.ModeLocal, "manual", 0, []string{"bssid:00:aa:bb:00:00:01"}, geo(50.087070, 14.420000, 20))
	// A third, ~5 km away, with a small circle: cannot be right as well.
	b.add(key(3), obs.ModeLocal, "manual", 0, []string{"bssid:00:aa:bb:00:00:02"}, geo(50.0870, 14.4200, 20))
	b.add(key(4), obs.ModeLocal, "manual", 0, []string{"bssid:00:aa:bb:00:00:02"}, geo(50.1320, 14.4200, 20))
	g := Build(b.list)

	ok := byIdent(t, g, "bssid:00:aa:bb:00:00:01").Loc
	if ok.Conflict || ok.Level != LevelPrecise {
		t.Fatalf("agreeing estimates: %+v", ok)
	}
	if ok.RadiusM >= 20 || math.Abs(ok.RadiusM-20/math.Sqrt2) > 0.5 {
		t.Fatalf("two agreeing 20 m estimates should give ≈14 m, got %.1f", ok.RadiusM)
	}
	if math.Abs(ok.Point.Lat-50.087035) > 0.00001 {
		t.Fatalf("position should sit between the two: %+v", ok.Point)
	}
	if ok.Conf <= 0.99 {
		t.Fatalf("combined confidence %.3f", ok.Conf)
	}

	bad := byIdent(t, g, "bssid:00:aa:bb:00:00:02").Loc
	if !bad.Conflict || math.Abs(bad.Conf-0.6) > 0.01 {
		t.Fatalf("incompatible estimates must be flagged and cost confidence: %+v", bad)
	}
	if len(bad.Sources) != 2 {
		t.Fatalf("both sources stay visible: %+v", bad.Sources)
	}
}

func TestLocationDecay(t *testing.T) {
	b := &builder{t: t}
	obsr := "rkm:aa"
	// A GPS fix is about a moment: after 2 h its confidence halves, after 10 h it is gone.
	b.add(key(1), obs.ModeLocal, "gpsd", 0, []string{obsr, "mac:00:11:32:00:00:99"}, geo(50.0, 14.0, 8))
	b.add(key(1), obs.ModeLocal, "self", 2*time.Hour, []string{obsr}, obs.Claim{K: "hostname", V: "laptop"})
	if c := byIdent(t, Build(b.list), obsr).Loc.Conf; math.Abs(c-0.5) > 0.01 {
		t.Fatalf("after one half-life: %.3f", c)
	}
	b.add(key(1), obs.ModeLocal, "self", 10*time.Hour, []string{obsr}, obs.Claim{K: "hostname", V: "laptop"})
	if l := byIdent(t, Build(b.list), obsr).Loc; l.Level != LevelUnknown || l.Point != nil {
		t.Fatalf("a stale fix must expire: %+v", l)
	}
}

func TestLocationLogicalAndRegion(t *testing.T) {
	b := &builder{t: t}
	A := key(1)
	// sysLocation of a switch is a place without coordinates.
	b.add(A, obs.ModeActive, "snmp", 0, []string{swMAC}, obs.Claim{K: "location.text", V: "Building A, room 101", C: 600})
	b.add(A, obs.ModeActive, "snmp", 0, []string{hostMAC}, obs.Claim{K: "link.l2", V: swMAC})
	b.add(A, obs.ModeActive, "snmp", 0, []string{farMAC}, obs.Claim{K: "link.l2-behind", V: swMAC})
	b.add(A, obs.ModeLocal, "manual", 0, []string{rtrMAC}, obs.Claim{K: "loc.region", V: "Praha, CZ"})
	b.add(A, obs.ModeLocal, "wifi-assoc", 0, []string{"bssid:00:aa:bb:00:00:01"}, obs.Claim{K: "link.wifi", V: swMAC})
	g := Build(b.list)

	sw := byIdent(t, g, swMAC)
	if l := sw.Loc; l.Level != LevelUnknown || l.Logical != "Building A, room 101" || l.LogicalFrom != "" {
		t.Fatalf("logical-only switch: %+v", l)
	}
	if l := byIdent(t, g, hostMAC).Loc; l.Logical != "Building A, room 101" || l.LogicalFrom != sw.ID {
		t.Fatalf("room should carry over a cable: %+v", l)
	}
	if l := byIdent(t, g, "bssid:00:aa:bb:00:00:01").Loc; l.Logical != "" {
		t.Fatalf("a radio link spans rooms, it must not carry a room: %+v", l)
	}
	if l := byIdent(t, g, farMAC).Loc; l.Logical != "" {
		t.Fatalf("room must not carry to 'somewhere behind': %+v", l)
	}
	if l := byIdent(t, g, rtrMAC).Loc; l.Level != LevelRegion || l.Point != nil || l.Region != "Praha, CZ" {
		t.Fatalf("region only: %+v", l)
	}
}

func TestLocationBadValuesIgnored(t *testing.T) {
	b := &builder{t: t}
	for _, v := range []string{"91,14,10", "50,181,10", "50,14,0", "50,14", "nan,14,10", "50,14,-5", "x"} {
		b.add(key(1), obs.ModeLocal, "manual", 0, []string{swMAC}, obs.Claim{K: "loc.geo", V: v})
	}
	if l := byIdent(t, Build(b.list), swMAC).Loc; l.Level != LevelUnknown {
		t.Fatalf("invalid coordinates must be ignored: %+v", l)
	}
}

// Merging maps from several observers in any order gives the same position.
func TestLocationOrderIndependent(t *testing.T) {
	b := &builder{t: t}
	switchWithNeighbours(b)
	b.add(key(2), obs.ModeLocal, "manual", time.Minute, []string{swMAC}, geo(50.08703, 14.42003, 15))
	b.add(key(3), obs.ModeLocal, "gpsd", 2*time.Minute, []string{swMAC}, geo(50.08697, 14.41998, 12))
	ref := Build(b.list)
	for range 20 {
		sh := append([]*obs.Observation(nil), b.list...)
		rand.Shuffle(len(sh), func(i, j int) { sh[i], sh[j] = sh[j], sh[i] })
		g := Build(sh)
		for i, d := range ref.Devices {
			a, c := d.Loc, g.Devices[i].Loc
			if a.Level != c.Level || a.RadiusM != c.RadiusM || a.Conf != c.Conf || a.Conflict != c.Conflict ||
				(a.Point == nil) != (c.Point == nil) || (a.Point != nil && *a.Point != *c.Point) {
				t.Fatalf("location depends on input order:\n%+v\n%+v", a, c)
			}
		}
	}
}
