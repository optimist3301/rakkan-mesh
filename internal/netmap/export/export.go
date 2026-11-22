// Package export renders a resolved graph. These are output-only views; the
// mergeable format is .rkm (package rkm).
package export

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

func JSON(w io.Writer, g *resolve.Graph) error {
	e := json.NewEncoder(w)
	e.SetIndent("", "  ")
	return e.Encode(g)
}

func ago(ms int64) string {
	d := time.Since(time.UnixMilli(ms)).Round(time.Second)
	switch {
	case d < time.Minute:
		return "now"
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
	return fmt.Sprintf("%dd", int(d.Hours()/24))
}

func netTitle(n *resolve.Network) string {
	t := n.Best("prefix")
	if t == "" {
		t = "?"
	}
	if m := n.Best("medium"); m != "" {
		t += " (" + m + ")"
	}
	if n.Scoped {
		t += " [observer-local id]"
	}
	return t
}

func ipsOf(d *resolve.Device) string {
	var v4, v6 []string
	for _, a := range d.Addrs {
		if strings.Contains(a.IP, ":") {
			v6 = append(v6, a.IP)
		} else {
			v4 = append(v4, a.IP)
		}
	}
	s := strings.Join(v4, ",")
	if len(v6) > 0 {
		if s != "" {
			s += " "
		}
		s += fmt.Sprintf("+%d v6", len(v6))
	}
	return s
}

func macOf(d *resolve.Device) string {
	for _, id := range d.Idents {
		if k := obs.Kind(id); k == "mac" || k == "lmac" {
			m := obs.Value(id)
			if k == "lmac" {
				m += "*"
			}
			return m
		}
	}
	return ""
}

func kindsOf(d *resolve.Device) string {
	var ks []string
	for _, id := range d.Idents {
		k := obs.Kind(id)
		if !slices.Contains(ks, k) {
			ks = append(ks, k)
		}
	}
	return strings.Join(ks, ",")
}

func info(d *resolve.Device) string {
	var parts []string
	if d.IsSelf {
		parts = append(parts, "★ this observer")
	}
	for _, k := range []string{"role", "vendor", "model", "os.family"} {
		if v := d.Best(k); v != "" && v != "observer" {
			parts = append(parts, v)
		}
	}
	var svcs []string
	for _, k := range []string{"port", "service"} {
		for _, v := range d.Claims[k] {
			svcs = append(svcs, strings.TrimPrefix(strings.TrimPrefix(v.Value, "tcp/"), "mdns/"))
		}
	}
	if len(svcs) > 0 {
		if len(svcs) > 6 {
			svcs = append(svcs[:6], "…")
		}
		parts = append(parts, "["+strings.Join(svcs, " ")+"]")
	}
	return strings.Join(parts, " ")
}

// Text prints a human-readable summary grouped by network.
func Text(w io.Writer, g *resolve.Graph) {
	fmt.Fprintf(w, "%d observations from %d observer(s) → %d networks, %d devices, %d links\n",
		g.Observations, len(g.Observers), len(g.Networks), len(g.Devices), len(g.Links))
	if len(g.ObserverInfo) > 0 {
		counts := map[string]int{}
		ignored := 0
		for _, i := range g.ObserverInfo {
			counts[i.Level.String()]++
			ignored += i.Ignored
		}
		var parts []string
		for _, l := range []string{"self", "trusted", "unknown", "blocked"} {
			if counts[l] > 0 {
				parts = append(parts, fmt.Sprintf("%d %s", counts[l], l))
			}
		}
		fmt.Fprintf(w, "observers: %s", strings.Join(parts, ", "))
		if ignored > 0 {
			fmt.Fprintf(w, "; %d observations ignored (blocked or revoked keys)", ignored)
		}
		if counts["unknown"] > 0 {
			fmt.Fprint(w, " — unknown observers count half (rakkan trust)")
		}
		fmt.Fprintln(w)
	}
	shown := map[string]bool{}
	byID := map[string]*resolve.Device{}
	for _, d := range g.Devices {
		byID[d.ID] = d
	}
	printDevs := func(ids []string) {
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "  ID\tNAME\tTYPE\tIP\tMAC\tLOCATION\tSEEN\tMODES\tINFO")
		for _, d := range g.Devices {
			if !slices.Contains(ids, d.ID) {
				continue
			}
			shown[d.ID] = true
			fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", d.ID[:7], trunc(d.Label, 28), classShort(d), ipsOf(d), macOf(d),
				locShort(d), ago(d.Last), strings.Join(d.Modes, "+"), info(d))
		}
		tw.Flush()
	}
	for _, n := range g.Networks {
		ssid := ""
		if v := n.Best("wifi.ssid"); v != "" {
			ssid = "  ssid " + v
		}
		wan := ""
		for _, k := range []string{"wan.stun", "wan.nat-pmp", "wan.upnp"} {
			if v := n.Best(k); v != "" {
				wan = "  wan " + v
				break
			}
		}
		if n.AS != nil {
			wan += fmt.Sprintf(" (AS%d %s)", n.AS.Number, n.AS.Name)
		}
		fmt.Fprintf(w, "\nnetwork %s  %s  gw %s%s%s  — %d devices\n", n.ID, netTitle(n), n.Best("gateway"), ssid, wan, len(n.Devices))
		printDevs(n.Devices)
	}
	var aps []*resolve.Device
	for _, d := range g.Devices {
		if !shown[d.ID] && d.Best("role") == "access-point" {
			aps = append(aps, d)
			shown[d.ID] = true
		}
	}
	if len(aps) > 0 {
		wifiAPs(w, aps)
	}
	if len(g.ASes) > 0 {
		fmt.Fprintln(w, "\ninternet (autonomous systems):")
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		for _, gr := range g.ASes {
			var names []string
			for _, id := range gr.Devices {
				shown[id] = true
				if d := byID[id]; d != nil {
					names = append(names, d.Label)
				}
			}
			if len(names) > 6 {
				names = append(names[:6], "…")
			}
			nets := ""
			if len(gr.Networks) > 0 {
				nets = fmt.Sprintf("%d network(s) exit here; ", len(gr.Networks))
			}
			fmt.Fprintf(tw, "  AS%d\t%s\t%s\t%s%s\n", gr.Number, trunc(gr.Name, 30), gr.Country, nets, strings.Join(names, ", "))
		}
		tw.Flush()
	}
	var rest []string
	for _, d := range g.Devices {
		if !shown[d.ID] {
			rest = append(rest, d.ID)
		}
	}
	if len(rest) > 0 {
		fmt.Fprintln(w, "\nrouted / not attached to a known network:")
		printDevs(rest)
	}
	if len(g.Links) > 0 {
		fmt.Fprintln(w, "\nlinks (l2 = cable, l2-behind = somewhere behind that port, l3 = routed hop):")
		for _, l := range g.Links {
			fmt.Fprintf(w, "  %s —%s— %s  %s\n", byID[l.A].Label, l.Kind, byID[l.B].Label, l.Detail)
		}
	}
	if len(g.Conflicts) > 0 {
		fmt.Fprintf(w, "\n%d identity conflict(s) refused (see `rakkan show --json`)\n", len(g.Conflicts))
	}
	fmt.Fprintln(w, "\nMAC* = random/virtual MAC, unique only within its network")
	fmt.Fprintln(w, "TYPE: inferred from names, services, ports and roles; ? = below 60% confidence")
	fmt.Fprintln(w, "LOCATION: precise/area/region ±radius; ~ = carried over links, not observed; ! = sources disagree")
}

// wifiAPs lists access points in range, strongest signal first.
func wifiAPs(w io.Writer, aps []*resolve.Device) {
	rssi := func(d *resolve.Device) int {
		v, err := strconv.Atoi(d.Best("rssi.dbm"))
		if err != nil {
			return -999
		}
		return v
	}
	slices.SortStableFunc(aps, func(a, b *resolve.Device) int { return rssi(b) - rssi(a) })
	fmt.Fprintf(w, "\nWi-Fi access points in range — %d\n", len(aps))
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  SSID\tBSSID\tBAND\tCH\tSECURITY\tRSSI\tSEEN\tOBSERVERS")
	for _, d := range aps {
		ssid := d.Best("ssid")
		if ssid == "" {
			ssid = "<hidden>"
		}
		bssid := ""
		for _, id := range d.Idents {
			if obs.Kind(id) == "bssid" {
				bssid = obs.Value(id)
				break
			}
		}
		r := d.Best("rssi.dbm")
		if r != "" {
			r += " dBm"
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n", trunc(ssid, 28), bssid, d.Best("wifi.band"),
			d.Best("wifi.channel"), d.Best("wifi.security"), r, ago(d.Last), len(d.Observers))
	}
	tw.Flush()
}

// locShort renders a device's location for the table.
func locShort(d *resolve.Device) string {
	l := d.Loc
	if l == nil {
		return "-"
	}
	var s string
	switch {
	case l.Point != nil:
		s = fmt.Sprintf("%s ±%s", l.Level, meters(l.RadiusM))
		if l.Derived {
			s += "~"
		}
		if l.Conflict {
			s += "!"
		}
	case l.Level == resolve.LevelRegion:
		s = "region"
	default:
		s = "unknown"
	}
	switch {
	case l.Logical != "":
		s += " “" + trunc(l.Logical, 22) + "”"
	case l.Region != "":
		s += " " + trunc(l.Region, 22)
	}
	return s
}

func meters(m float64) string {
	if m >= 1000 {
		return fmt.Sprintf("%.1fkm", m/1000)
	}
	return fmt.Sprintf("%.0fm", m)
}

func explainLocation(w io.Writer, d *resolve.Device, name func(string) string) {
	l := d.Loc
	if l == nil || (l.Level == resolve.LevelUnknown && l.Logical == "") {
		fmt.Fprintln(w, "\nlocation: unknown")
		return
	}
	fmt.Fprintf(w, "\nlocation: %s", l.Level)
	if l.Point != nil {
		how := "observed"
		if l.Derived {
			how = "carried over links, nobody observed it here"
		}
		fmt.Fprintf(w, "  %.6f, %.6f  ±%s  confidence %.0f%%  (%s)", l.Point.Lat, l.Point.Lon, meters(l.RadiusM), l.Conf*100, how)
		if l.Conflict {
			fmt.Fprint(w, "\n  ! sources disagree: the most precise trustworthy one is used, confidence is reduced")
		}
	}
	if l.Region != "" {
		fmt.Fprintf(w, "\n  region: %s", l.Region)
	}
	if l.Logical != "" {
		fmt.Fprintf(w, "\n  place: %s", l.Logical)
		if l.LogicalFrom != "" {
			fmt.Fprintf(w, "  (from %s, over a link)", name(l.LogicalFrom))
		}
	}
	fmt.Fprintln(w)
	for _, src := range l.Sources {
		if src.Kind == "link" {
			fmt.Fprintf(w, "  · from %s via %s: %.6f, %.6f ±%s  %.0f%%\n", name(src.From), strings.Join(src.Via, "→"),
				src.Point.Lat, src.Point.Lon, meters(src.RadiusM), src.Conf*100)
		} else {
			fmt.Fprintf(w, "  · %s by %s: %.6f, %.6f ±%s  %.0f%%  %s\n", src.Method, src.Observer,
				src.Point.Lat, src.Point.Lon, meters(src.RadiusM), src.Conf*100, ago(src.Time))
		}
	}
}

// GeoJSON writes located devices as a FeatureCollection (points; the error
// radius and level are properties).
func GeoJSON(w io.Writer, g *resolve.Graph) error {
	type feature struct {
		Type       string         `json:"type"`
		Geometry   map[string]any `json:"geometry"`
		Properties map[string]any `json:"properties"`
	}
	fc := struct {
		Type     string    `json:"type"`
		Features []feature `json:"features"`
	}{Type: "FeatureCollection", Features: []feature{}}
	for _, d := range g.Devices {
		l := d.Loc
		if l == nil || l.Point == nil {
			continue
		}
		props := map[string]any{"id": d.ID, "label": d.Label, "level": l.Level.String(), "radius_m": math.Round(l.RadiusM*10) / 10,
			"confidence": math.Round(l.Conf*1000) / 1000, "derived": l.Derived, "conflict": l.Conflict}
		if l.Logical != "" {
			props["place"] = l.Logical
		}
		fc.Features = append(fc.Features, feature{"Feature",
			map[string]any{"type": "Point", "coordinates": []float64{l.Point.Lon, l.Point.Lat}}, props}) // GeoJSON is lon,lat
	}
	e := json.NewEncoder(w)
	e.SetIndent("", "  ")
	return e.Encode(fc)
}

// classShort is the TYPE column: the class, with "?" below 60% confidence.
func classShort(d *resolve.Device) string {
	c := d.Class
	switch {
	case c == nil:
		return "-"
	case c.Conf < 0.6:
		return c.Kind + "?"
	}
	return c.Kind
}

func explainClass(w io.Writer, d *resolve.Device) {
	c := d.Class
	if c == nil {
		fmt.Fprintln(w, "\ntype: unknown (no telling evidence)")
		return
	}
	fmt.Fprintf(w, "\ntype: %s  confidence %.0f%%", c.Kind, c.Conf*100)
	if c.Alt != "" {
		fmt.Fprintf(w, "  (or %s %.0f%%)", c.Alt, c.AltConf*100)
	}
	fmt.Fprintln(w)
	for _, r := range c.Reasons {
		fmt.Fprintf(w, "  · %s\n", r)
	}
}

// Explain prints everything known about one device and why it was merged.
func Explain(w io.Writer, g *resolve.Graph, d *resolve.Device) {
	fmt.Fprintf(w, "device %s  %q\n", d.ID, d.Label)
	fmt.Fprintf(w, "  seen %s … %s, %d observations, observers %v, modes %v\n",
		time.UnixMilli(d.First).Format(time.DateTime), time.UnixMilli(d.Last).Format(time.DateTime), d.Obs, d.Observers, d.Modes)
	fmt.Fprintf(w, "  methods: %s\n  networks: %s\n\nidentifiers:\n", strings.Join(d.Methods, ", "), strings.Join(d.Networks, ", "))
	for _, id := range d.Idents {
		fmt.Fprintf(w, "  %-7s %s\n", []string{"weak", "medium", "strong"}[obs.StrengthOf(id)], id)
	}
	if len(d.Addrs) > 0 {
		fmt.Fprintln(w, "\naddresses:")
		for _, a := range d.Addrs {
			fmt.Fprintf(w, "  %-28s last %s ago\n", a.IP, ago(a.Last))
		}
	}
	explainClass(w, d)
	explainLocation(w, d, func(id string) string {
		if o := g.Device(id); o != nil {
			return o.Label
		}
		return id[:min(7, len(id))]
	})
	fmt.Fprintln(w, "\nclaims:")
	keys := make([]string, 0, len(d.Claims))
	for k := range d.Claims {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, k := range keys {
		for _, v := range d.Claims[k] {
			fmt.Fprintf(tw, "  %s\t%s\t%.0f%%\t×%d\t%s\n", k, trunc(v.Value, 60), float64(v.Conf)/10, v.Count, strings.Join(v.Modes, "+"))
		}
	}
	tw.Flush()
	if len(d.Evidence) > 0 {
		fmt.Fprintln(w, "\nwhy these identifiers are one device:")
		for _, e := range d.Evidence {
			fmt.Fprintf(w, "  %s ≡ %s  [%s: %s]\n", e.A, e.B, e.Rule, e.Detail)
		}
	}
}

func trunc(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

// DOT renders Graphviz: networks as boxes, devices as ellipses.
func DOT(w io.Writer, g *resolve.Graph) {
	q := func(s string) string { return fmt.Sprintf("%q", s) }
	fmt.Fprintln(w, "graph rakkan {\n  overlap=false; splines=true; node [fontname=\"sans\" fontsize=10];")
	for _, n := range g.Networks {
		fmt.Fprintf(w, "  %s [shape=box style=filled fillcolor=\"#dde8f5\" label=%s];\n", q(n.ID), q(netTitle(n)+"\n"+n.ID))
	}
	for _, d := range g.Devices {
		lbl := d.Label
		if ip := ipsOf(d); ip != "" && ip != d.Label {
			lbl += "\n" + ip
		}
		if v := d.Best("model"); v != "" {
			lbl += "\n" + trunc(v, 30)
		}
		style := ""
		switch {
		case d.IsSelf:
			style = ` style=filled fillcolor="#ffe9a8"`
		case d.Best("role") == "gateway" || d.Best("role") == "router":
			style = ` shape=diamond style=filled fillcolor="#f5d5d5"`
		}
		fmt.Fprintf(w, "  %s [label=%s%s];\n", q(d.ID), q(lbl), style)
		for _, n := range d.Networks {
			fmt.Fprintf(w, "  %s -- %s [color=\"#9aaabb\"];\n", q(d.ID), q(n))
		}
	}
	for _, l := range g.Links {
		style := `penwidth=2`
		switch l.Kind {
		case "l2-behind":
			style = `style=dashed color="#888888"`
		case "l3":
			style = `penwidth=1.5 color="#3366cc"`
		}
		fmt.Fprintf(w, "  %s -- %s [%s label=%s];\n", q(l.A), q(l.B), style, q(l.Detail))
	}
	fmt.Fprintln(w, "}")
}
