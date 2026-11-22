// Package share decides what leaves this machine when a map is exported or
// synced. Observations are signed, so they cannot be edited: an
// observation signed by someone else is either passed on intact or not at
// all. Only this installation's own observations can be redacted, by
// re-signing a reduced copy with the same key (spec §10.3).
package share

import (
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

// Who selects observers whose observations may be shared.
type Who int

const (
	All     Who = iota // everything except blocked keys
	Trusted            // own and trusted keys
	Own                // own key only
)

func ParseWho(s string) (Who, error) {
	switch strings.ToLower(s) {
	case "", "all":
		return All, nil
	case "trusted":
		return Trusted, nil
	case "own":
		return Own, nil
	}
	return All, fmt.Errorf("unknown --only %q (all, trusted, own)", s)
}

// Location says what happens to positions and place names.
type Location struct {
	Drop    bool    // remove positions, places and regions
	CoarseM float64 // >0: snap positions to a grid of this size, remove room-level places
}

// ParseLocation reads "keep", "drop" or "coarse:1km" / "coarse:500m".
func ParseLocation(s string) (Location, error) {
	switch {
	case s == "" || s == "keep":
		return Location{}, nil
	case s == "drop":
		return Location{Drop: true}, nil
	case strings.HasPrefix(s, "coarse:"):
		v := strings.TrimPrefix(s, "coarse:")
		mult := 1.0
		switch {
		case strings.HasSuffix(v, "km"):
			v, mult = strings.TrimSuffix(v, "km"), 1000
		case strings.HasSuffix(v, "m"):
			v = strings.TrimSuffix(v, "m")
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f <= 0 || math.IsInf(f, 0) {
			return Location{}, fmt.Errorf("bad grid in %q", s)
		}
		return Location{CoarseM: f * mult}, nil
	}
	return Location{}, fmt.Errorf("unknown --location %q (keep, drop, coarse:1km)", s)
}

type Policy struct {
	Who      Who
	Since    int64 // unix ms; 0 = no limit
	Location Location
	DropWAN  bool // withhold external addresses of networks (method "wan")
}

// Report says what the policy did, for the user.
type Report struct {
	Kept, Redacted int
	Dropped        map[string]int // reason → count
}

func (r Report) String() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("%d shared", r.Kept+r.Redacted))
	if r.Redacted > 0 {
		parts = append(parts, fmt.Sprintf("%d of your own redacted and re-signed", r.Redacted))
	}
	reasons := make([]string, 0, len(r.Dropped))
	for k := range r.Dropped {
		reasons = append(reasons, k)
	}
	slices.Sort(reasons)
	for _, k := range reasons {
		parts = append(parts, fmt.Sprintf("%d withheld (%s)", r.Dropped[k], k))
	}
	return strings.Join(parts, ", ")
}

var locKeys = map[string]bool{"loc.geo": true, "loc.logical": true, "location.text": true, "loc.region": true}

func sensitive(o *obs.Observation, l Location) bool {
	if !l.Drop && l.CoarseM == 0 {
		return false
	}
	for _, c := range o.Claims {
		if locKeys[c.K] {
			return true
		}
	}
	return false
}

// redactClaims applies the location policy to a claim list.
func redactClaims(cs []obs.Claim, l Location) []obs.Claim {
	out := make([]obs.Claim, 0, len(cs))
	for _, c := range cs {
		switch {
		case !locKeys[c.K]:
			out = append(out, c)
		case l.Drop:
		case c.K == "loc.region":
			out = append(out, c) // a country or city is coarse already
		case c.K == "loc.geo":
			if lat, lon, r, err := obs.ParseGeo(c.V); err == nil {
				lat, lon, r = Coarsen(lat, lon, r, l.CoarseM)
				c.V = obs.FormatGeo(lat, lon, r)
				out = append(out, c)
			}
		default: // rooms and floors are finer than any useful grid
		}
	}
	return out
}

// Coarsen snaps a position to the centre of its grid cell and widens the
// radius so the original error circle stays inside the new one.
func Coarsen(lat, lon, r, gridM float64) (float64, float64, float64) {
	const mPerDeg = 111320.0
	dLat := gridM / mPerDeg
	lat = (math.Floor(lat/dLat) + 0.5) * dLat
	lat = math.Max(-90, math.Min(90, lat))
	dLon := gridM / (mPerDeg * math.Max(0.01, math.Cos(lat*math.Pi/180)))
	lon = (math.Floor(lon/dLon) + 0.5) * dLon
	lon = math.Max(-180, math.Min(180, lon))
	// The true point may sit in any corner of the cell, and r further still.
	return lat, lon, r + gridM*math.Sqrt2/2
}

// Apply returns what may be shared. key is this installation's key (used
// to recognise and re-sign own observations); book may be nil.
func Apply(list []*obs.Observation, p Policy, key ed25519.PrivateKey, book *trust.Book) ([]*obs.Observation, Report, error) {
	own := hex.EncodeToString(key.Public().(ed25519.PublicKey))
	rep := Report{Dropped: map[string]int{}}
	var out []*obs.Observation
	for _, o := range list {
		k := hex.EncodeToString(o.Observer)
		level := trust.Unknown
		if k == own {
			level = trust.Self
		} else if book != nil {
			level = book.Level(k)
		}
		switch {
		case level == trust.Blocked:
			rep.Dropped["blocked observer"]++
			continue
		case p.Who == Own && level != trust.Self, p.Who == Trusted && level < trust.Trusted:
			rep.Dropped["observer not selected"]++
			continue
		case p.Since > 0 && max(o.Time, o.Until) < p.Since:
			rep.Dropped["too old"]++
			continue
		case p.DropWAN && o.Method == "wan":
			rep.Dropped["external address"]++
			continue
		case o.V != obs.SchemaVersion && (p.Location.Drop || p.Location.CoarseM > 0 || p.DropWAN):
			rep.Dropped["unreadable newer format"]++ // cannot check it for positions
			continue
		}
		if !sensitive(o, p.Location) {
			out = append(out, o)
			rep.Kept++
			continue
		}
		if k != own {
			rep.Dropped["someone else's location"]++
			continue
		}
		b := o.Body
		b.Claims = redactClaims(o.Claims, p.Location)
		if len(b.Claims) == 0 {
			rep.Dropped["location only"]++
			continue
		}
		red, err := obs.Sign(b, key)
		if err != nil {
			return nil, rep, err
		}
		out = append(out, red)
		rep.Redacted++
	}
	return out, rep, nil
}
