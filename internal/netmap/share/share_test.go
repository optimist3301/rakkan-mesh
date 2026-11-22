package share

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

func key(b byte) ed25519.PrivateKey { return ed25519.NewKeyFromSeed(bytes.Repeat([]byte{b}, 32)) }
func hexKey(k ed25519.PrivateKey) string {
	return hex.EncodeToString(k.Public().(ed25519.PublicKey))
}

func sign(t *testing.T, k ed25519.PrivateKey, ts int64, subject string, claims ...obs.Claim) *obs.Observation {
	t.Helper()
	o, err := obs.Sign(obs.Body{Time: ts, Mode: obs.ModeLocal, Method: "manual", Subject: []string{subject}, Claims: claims}, k)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

func geo(lat, lon, r float64) obs.Claim {
	return obs.Claim{K: "loc.geo", V: obs.FormatGeo(lat, lon, r)}
}

func TestParse(t *testing.T) {
	for in, want := range map[string]float64{"coarse:1km": 1000, "coarse:250m": 250, "coarse:2.5km": 2500, "coarse:300": 300} {
		l, err := ParseLocation(in)
		if err != nil || l.CoarseM != want {
			t.Errorf("%s → %+v %v", in, l, err)
		}
	}
	for _, bad := range []string{"coarse:", "coarse:-1km", "coarse:xkm", "blur", "coarse:Infkm"} {
		if _, err := ParseLocation(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if _, err := ParseWho("friends"); err == nil {
		t.Error("bad --only accepted")
	}
}

func TestCoarsenKeepsTruthInsideCircle(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for range 2000 {
		lat, lon := r.Float64()*170-85, r.Float64()*358-179
		radius := r.Float64() * 50
		grid := 100 + r.Float64()*5000
		clat, clon, cr := Coarsen(lat, lon, radius, grid)
		if d := distanceM(lat, lon, clat, clon); d+radius > cr+1e-6 {
			t.Fatalf("(%f,%f ±%.0f) grid %.0f → (%f,%f ±%.0f): original circle sticks out by %.1f m",
				lat, lon, radius, grid, clat, clon, cr, d+radius-cr)
		}
		// Every point of a cell maps to the same coarse value: that is what hides it.
		a1, o1, _ := Coarsen(clat, clon, 0, grid)
		if math.Abs(a1-clat) > 1e-9 || math.Abs(o1-clon) > 1e-6 {
			t.Fatalf("coarsening is not idempotent at (%f,%f)", lat, lon)
		}
	}
}

func distanceM(lat1, lon1, lat2, lon2 float64) float64 {
	rad := math.Pi / 180
	dLat, dLon := (lat2-lat1)*rad, (lon2-lon1)*rad
	h := math.Sin(dLat/2)*math.Sin(dLat/2) + math.Cos(lat1*rad)*math.Cos(lat2*rad)*math.Sin(dLon/2)*math.Sin(dLon/2)
	return 2 * 6371008.8 * math.Asin(math.Min(1, math.Sqrt(h)))
}

func TestApply(t *testing.T) {
	me, friend, stranger, troll := key(1), key(2), key(3), key(4)
	book := trust.NewBook(trust.Entry{Key: hexKey(friend), Level: trust.Trusted},
		trust.Entry{Key: hexKey(troll), Level: trust.Blocked})
	list := []*obs.Observation{
		sign(t, me, 1000, "mac:00:11:32:00:00:01", obs.Claim{K: "name", V: "nas"}),
		sign(t, me, 1000, "rkm:aa", geo(50.0871, 14.4201, 10), obs.Claim{K: "loc.logical", V: "Lab / desk 3"}),
		sign(t, me, 1000, "mac:00:11:32:00:00:02", obs.Claim{K: "name", V: "sw"}, obs.Claim{K: "location.text", V: "Room 101"}),
		sign(t, friend, 1000, "mac:00:11:32:00:00:03", obs.Claim{K: "name", V: "printer"}),
		sign(t, friend, 1000, "mac:00:11:32:00:00:04", geo(50.0, 14.0, 5)),
		sign(t, stranger, 1000, "mac:00:11:32:00:00:05", obs.Claim{K: "name", V: "tv"}),
		sign(t, troll, 1000, "mac:00:11:32:00:00:06", obs.Claim{K: "name", V: "junk"}),
		sign(t, me, 10, "mac:00:11:32:00:00:07", obs.Claim{K: "name", V: "old"}),
	}

	// Keep everything (but never blocked keys).
	out, rep, _ := Apply(list, Policy{}, me, book)
	if len(out) != 7 || rep.Dropped["blocked observer"] != 1 {
		t.Fatalf("keep: %d, %v", len(out), rep)
	}
	// Signatures pass through untouched.
	for _, o := range out {
		if _, err := obs.Decode(o.Raw, o.Sig); err != nil {
			t.Fatal(err)
		}
	}

	// Own only, recent only.
	out, rep, _ = Apply(list, Policy{Who: Own, Since: 500}, me, book)
	if len(out) != 3 || rep.Dropped["too old"] != 1 {
		t.Fatalf("own+since: %d %v", len(out), rep)
	}
	out, _, _ = Apply(list, Policy{Who: Trusted}, me, book)
	if len(out) != 6 {
		t.Fatalf("trusted: %d", len(out))
	}

	// Coarse: own positions re-signed on a grid, own rooms removed; someone
	// else's positions cannot be edited, so they are withheld.
	out, rep, err := Apply(list, Policy{Location: Location{CoarseM: 1000}}, me, book)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Redacted != 2 || rep.Dropped["someone else's location"] != 1 {
		t.Fatalf("coarse report %v", rep)
	}
	for _, o := range out {
		if _, err := obs.Decode(o.Raw, o.Sig); err != nil {
			t.Fatalf("redacted copy must be validly signed: %v", err)
		}
		for _, c := range o.Claims {
			switch c.K {
			case "loc.logical", "location.text":
				t.Fatalf("room-level place leaked: %+v", c)
			case "loc.geo":
				lat, lon, r, _ := obs.ParseGeo(c.V)
				if r < 700 || distanceM(lat, lon, 50.0871, 14.4201) > r {
					t.Fatalf("coarse position %v", c.V)
				}
				if hex.EncodeToString(o.Observer) != hexKey(me) {
					t.Fatal("re-signed under the wrong key")
				}
			}
		}
	}

	// Drop: positions gone, the rest of the observation survives.
	out, rep, _ = Apply(list, Policy{Location: Location{Drop: true}}, me, book)
	if rep.Dropped["location only"] != 1 || rep.Redacted != 1 {
		t.Fatalf("drop report %v", rep)
	}
	for _, o := range out {
		for _, c := range o.Claims {
			if locKeys[c.K] {
				t.Fatalf("location leaked with drop: %+v", c)
			}
		}
	}
}

func TestDropWAN(t *testing.T) {
	me, friend := key(1), key(2)
	wan, _ := obs.Sign(obs.Body{Time: 1, Mode: obs.ModeActive, Method: "wan", Subject: []string{"gip:81.2.69.160"},
		Claims: []obs.Claim{{K: "role", V: "wan-address"}}}, friend)
	other := sign(t, me, 1, "mac:00:11:32:00:00:01", obs.Claim{K: "name", V: "nas"})
	out, rep, _ := Apply([]*obs.Observation{wan, other}, Policy{DropWAN: true}, me, nil)
	if len(out) != 1 || out[0] != other || rep.Dropped["external address"] != 1 {
		t.Fatalf("%d %v", len(out), rep)
	}
}
