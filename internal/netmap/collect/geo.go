//go:build linux

package collect

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// GPSD takes one position fix of the observer from a running gpsd and
// records it as the observer's location. It only reads; the receiver is
// owned by gpsd.
type GPSD struct {
	Addr    string // default 127.0.0.1:2947
	Timeout time.Duration
}

func (GPSD) Name() string { return "gpsd" }

// Fix is a position fix with its estimated horizontal error.
type Fix struct {
	Lat, Lon, RadiusM float64
	Time              time.Time
	Mode              int // 2 = 2D, 3 = 3D
}

type tpv struct {
	Class string  `json:"class"`
	Mode  int     `json:"mode"`
	Lat   float64 `json:"lat"`
	Lon   float64 `json:"lon"`
	Epx   float64 `json:"epx"` // gpsd: 95% error, metres
	Epy   float64 `json:"epy"`
	Time  string  `json:"time"`
}

// fixFrom converts a TPV report; ok is false for reports without a fix.
func fixFrom(r tpv, now time.Time) (Fix, bool) {
	if r.Class != "TPV" || r.Mode < 2 {
		return Fix{}, false
	}
	if _, _, _, err := obs.ParseGeo(obs.FormatGeo(r.Lat, r.Lon, 1)); err != nil || (r.Lat == 0 && r.Lon == 0) {
		return Fix{}, false
	}
	f := Fix{Lat: r.Lat, Lon: r.Lon, Mode: r.Mode, Time: now}
	if t, err := time.Parse(time.RFC3339Nano, r.Time); err == nil {
		f.Time = t
	}
	switch {
	case r.Epx > 0 && r.Epy > 0:
		f.RadiusM = math.Max(5, math.Hypot(r.Epx, r.Epy)) // never claim better than a consumer receiver can
	case r.Mode >= 3:
		f.RadiusM = 25 // no error estimate: a typical outdoor 3D fix
	default:
		f.RadiusM = 100
	}
	return f, true
}

func (g GPSD) Fix(ctx context.Context) (Fix, error) {
	addr := g.Addr
	if addr == "" {
		addr = "127.0.0.1:2947"
	}
	timeout := g.Timeout
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", addr)
	if err != nil {
		return Fix{}, fmt.Errorf("gpsd not reachable at %s: %w", addr, err)
	}
	defer conn.Close()
	go func() { <-ctx.Done(); conn.Close() }()
	if _, err := fmt.Fprint(conn, "?WATCH={\"enable\":true,\"json\":true}\n"); err != nil {
		return Fix{}, err
	}
	sc := bufio.NewScanner(conn)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		var r tpv
		if json.Unmarshal(sc.Bytes(), &r) != nil {
			continue
		}
		if f, ok := fixFrom(r, time.Now()); ok {
			return f, nil
		}
	}
	if ctx.Err() != nil {
		return Fix{}, errors.New("gpsd has no position fix (is the receiver indoors?)")
	}
	return Fix{}, sc.Err()
}

func (g GPSD) Run(ctx context.Context, s *Session, _ []NetCtx) error {
	f, err := g.Fix(ctx)
	if err != nil {
		return err
	}
	s.Emit(obs.Body{Time: f.Time.UnixMilli(), Mode: obs.ModeLocal, Method: "gpsd",
		Subject: []string{obs.RKMID(s.Pub)},
		Claims:  []obs.Claim{claim("loc.geo", obs.FormatGeo(f.Lat, f.Lon, f.RadiusM))}})
	s.Log.Printf("gpsd: fix %.6f,%.6f ±%.0f m", f.Lat, f.Lon, f.RadiusM)
	return nil
}

// Place is a hand-entered location. Any part may be empty.
type Place struct {
	Lat, Lon, RadiusM float64 // RadiusM == 0: no coordinates
	Logical, Region   string
}

// EmitPlace records a hand-entered location for the entity named by subject.
func EmitPlace(s *Session, subject []string, p Place) error {
	var claims []obs.Claim
	if p.RadiusM != 0 {
		if _, _, _, err := obs.ParseGeo(obs.FormatGeo(p.Lat, p.Lon, p.RadiusM)); err != nil {
			return err
		}
		claims = append(claims, claim("loc.geo", obs.FormatGeo(p.Lat, p.Lon, p.RadiusM)))
	}
	if p.Logical != "" {
		claims = append(claims, claim("loc.logical", p.Logical))
	}
	if p.Region != "" {
		claims = append(claims, claim("loc.region", p.Region))
	}
	if len(claims) == 0 {
		return errors.New("nothing to record: give coordinates, --logical or --region")
	}
	s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "manual", Subject: subject, Claims: claims})
	return nil
}
