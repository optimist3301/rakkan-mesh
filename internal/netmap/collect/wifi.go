//go:build linux

package collect

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/godbus/dbus/v5"
	"github.com/mdlayher/wifi"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// WifiScan records the access points in range — the same list a phone or
// laptop shows in its Wi-Fi menu — and which one this host is connected to.
// It reads the operating system's scan cache. With Refresh it first asks
// NetworkManager for a fresh scan, exactly as opening the Wi-Fi menu does
// (mode active, since the radio then transmits probe requests); otherwise
// it only reads what the OS already collected (mode passive).
//
// Only access points are recorded: client devices of other networks are not
// visible in the scan cache and are not collected.
type WifiScan struct {
	Refresh bool
	Wait    time.Duration // how long to wait for a requested scan
	MaxAge  time.Duration // ignore cache entries older than this
}

func (WifiScan) Name() string { return "wifi" }

func (c WifiScan) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	cl, err := wifi.New()
	if err != nil {
		return fmt.Errorf("nl80211: %w", err)
	}
	defer cl.Close()
	ifis, err := cl.Interfaces()
	if err != nil {
		return err
	}
	for _, ifi := range ifis {
		if ifi.Type != wifi.InterfaceTypeStation || ifi.Name == "" {
			continue
		}
		mode := obs.ModePassive
		if c.Refresh {
			if err := requestScan(ctx, ifi.Name, c.Wait); err != nil {
				s.Log.Printf("wifi %s: fresh scan not available (%v), using the OS cache", ifi.Name, err)
			} else {
				mode = obs.ModeActive
			}
		}
		aps, err := cl.AccessPoints(ifi)
		if err != nil {
			s.Log.Printf("wifi %s: %v", ifi.Name, err)
			continue
		}
		var n *NetCtx
		for i := range nets {
			if nets[i].Iface.Name == ifi.Name {
				n = &nets[i]
			}
		}
		var stations []*wifi.StationInfo
		if n != nil {
			stations, _ = cl.StationInfo(ifi)
		}
		count := 0
		for _, ap := range aps {
			if ap.LastSeen > c.MaxAge || len(ap.BSSID) != 6 {
				continue
			}
			count++
			emitAP(s, n, ifi, mode, ap, stations)
		}
		s.Log.Printf("wifi %s: %d access points in range", ifi.Name, count)
	}
	return nil
}

func bssidID(b []byte) string {
	if len(b) != 6 {
		return ""
	}
	return "bssid:" + strings.ToLower(fmt.Sprintf("%02x:%02x:%02x:%02x:%02x:%02x", b[0], b[1], b[2], b[3], b[4], b[5]))
}

func emitAP(s *Session, n *NetCtx, ifi *wifi.Interface, mode obs.Mode, ap *wifi.BSS, stations []*wifi.StationInfo) {
	vantage := &obs.Vantage{Iface: ifi.Name}
	if n != nil {
		vantage = n.Vantage()
	}
	band, ch := freqChannel(ap.Frequency)
	claims := []obs.Claim{
		claim("role", "access-point"),
		claim("wifi.freq", strconv.Itoa(ap.Frequency)),
		claim("wifi.band", band),
		claim("wifi.security", securityOf(ap.RSN)),
	}
	if ch > 0 {
		claims = append(claims, claim("wifi.channel", strconv.Itoa(ch)))
	}
	if ssid := sanitize([]byte(ap.SSID)); ssid != "" {
		claims = append(claims, claim("ssid", ssid))
	} else {
		claims = append(claims, claim("wifi.hidden", "true"))
	}
	if ap.Signal != 0 {
		// Signal strength as heard by this observer: input for location (phase 5).
		claims = append(claims, claimC("rssi.dbm", quantizeRSSI(int(ap.Signal)/100), 500))
	}
	t := time.Now().Add(-ap.LastSeen).UnixMilli()
	id := bssidID(ap.BSSID)
	s.Emit(obs.Body{Time: t, Mode: mode, Method: "wifi-scan", Vantage: vantage, Subject: []string{id}, Claims: claims})

	if ap.Status != wifi.BSSStatusAssociated || n == nil {
		return
	}
	// Our own connection: the AP bridges into the network we are on.
	assoc := []obs.Claim{
		claim("link.wifi", n.MACID(n.Iface.HardwareAddr)),
		claim("link.port.b", n.Iface.Name),
	}
	for _, st := range stations {
		if st.HardwareAddr.String() == ap.BSSID.String() {
			if st.Signal != 0 {
				assoc = append(assoc, claimC("rssi.dbm", quantizeRSSI(st.Signal), 500))
			}
			if st.TransmitBitrate > 0 {
				assoc = append(assoc, claimC("wifi.txrate.mbps", strconv.Itoa(st.TransmitBitrate/1_000_000), 300))
			}
		}
	}
	s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "wifi-assoc", Vantage: n.Vantage(), Subject: []string{id}, Claims: assoc})
	netClaims := []obs.Claim{claim("wifi.bssid", strings.TrimPrefix(id, "bssid:"))}
	if ssid := sanitize([]byte(ap.SSID)); ssid != "" {
		netClaims = append(netClaims, claim("wifi.ssid", ssid))
	}
	s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "wifi-assoc", Vantage: n.Vantage(),
		Subject: []string{obs.NetEntityID(n.NetID)}, Claims: netClaims})
}

// requestScan asks NetworkManager to rescan (what the Wi-Fi menu does) and
// waits until the scan completes or the timeout passes.
func requestScan(ctx context.Context, iface string, wait time.Duration) error {
	conn, err := dbus.ConnectSystemBus()
	if err != nil {
		return err
	}
	defer conn.Close()
	nm := conn.Object("org.freedesktop.NetworkManager", "/org/freedesktop/NetworkManager")
	var dev dbus.ObjectPath
	if err := nm.CallWithContext(ctx, "org.freedesktop.NetworkManager.GetDeviceByIpIface", 0, iface).Store(&dev); err != nil {
		return err
	}
	d := conn.Object("org.freedesktop.NetworkManager", dev)
	const wl = "org.freedesktop.NetworkManager.Device.Wireless"
	lastScan := func() int64 {
		v, err := d.GetProperty(wl + ".LastScan")
		if err != nil {
			return -1
		}
		x, _ := v.Value().(int64)
		return x
	}
	before := lastScan()
	if err := d.CallWithContext(ctx, wl+".RequestScan", 0, map[string]dbus.Variant{}).Err; err != nil {
		return err
	}
	deadline := time.Now().Add(wait)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(300 * time.Millisecond):
		}
		if lastScan() != before {
			return nil
		}
	}
	return errors.New("scan did not complete in time")
}
