//go:build linux

package collect

import (
	"context"
	"fmt"
	"math"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

// fakeGPSD speaks enough of the gpsd protocol: greets, waits for ?WATCH,
// then reports a no-fix TPV, noise, and a real fix.
func fakeGPSD(t *testing.T) string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		fmt.Fprintln(c, `{"class":"VERSION","release":"3.25"}`)
		buf := make([]byte, 256)
		c.Read(buf)
		fmt.Fprintln(c, `{"class":"TPV","mode":1}`)
		fmt.Fprintln(c, `not json`)
		fmt.Fprintln(c, `{"class":"SKY","satellites":[]}`)
		fmt.Fprintln(c, `{"class":"TPV","mode":3,"lat":50.087,"lon":14.42,"epx":6.0,"epy":8.0,"time":"`+time.Now().UTC().Format("2006-01-02T15:04:05.000Z")+`"}`)
		time.Sleep(time.Second)
	}()
	return ln.Addr().String()
}

func TestGPSDFix(t *testing.T) {
	f, err := GPSD{Addr: fakeGPSD(t), Timeout: 3 * time.Second}.Fix(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.Lat != 50.087 || f.Lon != 14.42 || math.Abs(f.RadiusM-10) > 1e-9 || f.Time.Year() != 2026 {
		t.Fatalf("%+v", f)
	}
}

func TestGPSDNoFixTimesOut(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err == nil {
			fmt.Fprintln(c, `{"class":"TPV","mode":1}`)
			time.Sleep(2 * time.Second)
			c.Close()
		}
	}()
	_, err := GPSD{Addr: ln.Addr().String(), Timeout: 300 * time.Millisecond}.Fix(context.Background())
	if err == nil || !strings.Contains(err.Error(), "no position fix") {
		t.Fatalf("got %v", err)
	}
	if _, err := (GPSD{Addr: "127.0.0.1:1", Timeout: time.Second}).Fix(context.Background()); err == nil {
		t.Fatal("unreachable gpsd must be an error")
	}
}

func TestFixFromRejectsBogus(t *testing.T) {
	now := time.Now()
	for _, r := range []tpv{{Class: "TPV", Mode: 1}, {Class: "SKY", Mode: 3, Lat: 1, Lon: 1},
		{Class: "TPV", Mode: 3, Lat: 95, Lon: 1}, {Class: "TPV", Mode: 3}} {
		if _, ok := fixFrom(r, now); ok {
			t.Errorf("%+v accepted", r)
		}
	}
	f, _ := fixFrom(tpv{Class: "TPV", Mode: 2, Lat: 1, Lon: 1}, now)
	if f.RadiusM != 100 {
		t.Errorf("2D without error estimate: %.0f", f.RadiusM)
	}
	f, _ = fixFrom(tpv{Class: "TPV", Mode: 3, Lat: 1, Lon: 1, Epx: 0.5, Epy: 0.5}, now)
	if f.RadiusM != 5 {
		t.Errorf("error floor: %.1f", f.RadiusM)
	}
}

// The observer's own GPS fix and a hand-placed access point end up on the
// right devices and the AP, linked by Wi-Fi, gets a derived area.
func TestPlacesEndToEnd(t *testing.T) {
	s, _ := testSession()
	GPSD{Addr: fakeGPSD(t), Timeout: 3 * time.Second}.Run(context.Background(), s, nil)
	if err := EmitPlace(s, []string{"bssid:00:aa:bb:00:00:01"}, Place{Logical: "Cafe, 1st floor"}); err != nil {
		t.Fatal(err)
	}
	if err := EmitPlace(s, []string{"bssid:00:aa:bb:00:00:01"}, Place{}); err == nil {
		t.Fatal("an empty place must be refused")
	}
	if err := EmitPlace(s, []string{"x:y"}, Place{Lat: 99, Lon: 0, RadiusM: 5}); err == nil {
		t.Fatal("invalid coordinates must be refused")
	}
	g := resolve.Build(s.Flush())
	var self, ap *resolve.Device
	for _, d := range g.Devices {
		if d.Loc.Point != nil {
			self = d
		}
		if d.Loc.Logical != "" {
			ap = d
		}
	}
	if self == nil || self.Loc.Level != resolve.LevelPrecise || ap == nil {
		t.Fatalf("self=%v ap=%v", self, ap)
	}
}
