//go:build linux

package collect

import (
	"bytes"
	"crypto/ed25519"
	"io"
	"net"
	"testing"

	"github.com/mdlayher/wifi"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

func TestWifiHelpers(t *testing.T) {
	for _, c := range []struct {
		mhz  int
		band string
		ch   int
	}{{2412, "2.4GHz", 1}, {2484, "2.4GHz", 14}, {5180, "5GHz", 36}, {5955, "6GHz", 1}, {900, "", 0}} {
		if b, ch := freqChannel(c.mhz); b != c.band || ch != c.ch {
			t.Errorf("%d MHz: got %s/%d", c.mhz, b, ch)
		}
	}
	for in, want := range map[int]string{-67: "-69", -66: "-66", -1: "-3", 0: "0"} {
		if got := quantizeRSSI(in); got != want {
			t.Errorf("quantize(%d) = %s, want %s", in, got, want)
		}
	}
	if s := securityOf(wifi.RSNInfo{Version: 1, AKMs: []wifi.RSNAKM{wifi.RSNAkmPSK, wifi.RSNAkmSAE}}); s != "WPA2-PSK/WPA3-SAE" {
		t.Errorf("transition mode: %s", s)
	}
	if s := securityOf(wifi.RSNInfo{}); s != "open/legacy" {
		t.Errorf("no RSN: %s", s)
	}
	if sanitize([]byte("Cafe\x00\x07 Wi-Fi\xff")) != "Cafe Wi-Fi" {
		t.Error("sanitize")
	}
}

func bss(mac string, ssid string, status wifi.BSSStatus, mbm int32) *wifi.BSS {
	hw, _ := net.ParseMAC(mac)
	return &wifi.BSS{BSSID: hw, SSID: ssid, Frequency: 5180, Signal: mbm, Status: status,
		RSN: wifi.RSNInfo{Version: 1, AKMs: []wifi.RSNAKM{wifi.RSNAkmPSK}}}
}

// The AP we are connected to joins our network and links to our interface;
// other APs are only "in range".
func TestWifiAssociation(t *testing.T) {
	s, n := testSession()
	n.Iface.HardwareAddr, _ = net.ParseMAC("00:11:32:00:00:77")
	ifi := &wifi.Interface{Name: "eth0"}
	emitAP(s, n, ifi, obs.ModePassive, bss("00:aa:bb:00:00:01", "Home", wifi.BSSStatusAssociated, -5500), nil)
	emitAP(s, n, ifi, obs.ModePassive, bss("00:aa:bb:00:00:02", "Neighbor", wifi.BSSStatusNotAssociated, -8100), nil)
	emitAP(s, n, ifi, obs.ModePassive, bss("00:aa:bb:00:00:03", "", wifi.BSSStatusNotAssociated, -9000), nil)
	s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "self", Subject: []string{"rkm:aa", "mac:00:11:32:00:00:77"}})
	g := resolve.Build(s.Flush())

	var home, neighbor, hidden *resolve.Device
	for _, d := range g.Devices {
		switch d.Best("ssid") {
		case "Home":
			home = d
		case "Neighbor":
			neighbor = d
		case "":
			if d.Best("wifi.hidden") == "true" {
				hidden = d
			}
		}
	}
	if home == nil || neighbor == nil || hidden == nil {
		t.Fatalf("missing APs: home=%v neighbor=%v hidden=%v", home, neighbor, hidden)
	}
	if len(home.Networks) != 1 || len(neighbor.Networks) != 0 {
		t.Fatalf("membership: home %v, neighbor %v", home.Networks, neighbor.Networks)
	}
	if home.Best("rssi.dbm") != "-57" || home.Best("wifi.security") != "WPA2-PSK" || home.Best("wifi.channel") != "36" {
		t.Fatalf("claims %+v", home.Claims)
	}
	if len(g.Links) != 1 || g.Links[0].Kind != "wifi" || g.Links[0].A != home.ID {
		t.Fatalf("links %+v", g.Links)
	}
	if g.Networks[0].Best("wifi.ssid") != "Home" {
		t.Fatalf("network ssid: %+v", g.Networks[0].Claims)
	}
}

// Two observers hearing the same AP produce one device with both signal
// readings kept (per-observer RSSI is the raw input for location).
func TestWifiMergeAcrossObservers(t *testing.T) {
	sA, nA := testSession()
	sB := NewSession(ed25519.NewKeyFromSeed(bytes.Repeat([]byte{5}, 32)), io.Discard)
	ifi := &wifi.Interface{Name: "wlan0"}
	emitAP(sA, nA, ifi, obs.ModePassive, bss("00:aa:bb:00:00:09", "Office", wifi.BSSStatusNotAssociated, -4000), nil)
	emitAP(sB, nil, ifi, obs.ModeActive, bss("00:aa:bb:00:00:09", "Office", wifi.BSSStatusNotAssociated, -7500), nil)
	g := resolve.Build(append(sA.Flush(), sB.Flush()...))
	if len(g.Devices) != 1 {
		t.Fatalf("want 1 AP, got %d", len(g.Devices))
	}
	d := g.Devices[0]
	if len(d.Observers) != 2 || len(d.Claims["rssi.dbm"]) != 2 || len(d.Modes) != 2 {
		t.Fatalf("observers=%v rssi=%v modes=%v", d.Observers, d.Claims["rssi.dbm"], d.Modes)
	}
}
