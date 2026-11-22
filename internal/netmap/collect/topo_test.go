//go:build linux

package collect

import (
	"bytes"
	"crypto/ed25519"
	"encoding/binary"
	"io"
	"net"
	"net/netip"
	"strings"
	"testing"
	"unsafe"

	"github.com/gosnmp/gosnmp"
	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

type fakeWalker []gosnmp.SnmpPDU

func (f fakeWalker) walk(oid string) ([]gosnmp.SnmpPDU, error) {
	var out []gosnmp.SnmpPDU
	for _, p := range f {
		if strings.HasPrefix(p.Name, oid+".") {
			out = append(out, p)
		}
	}
	return out, nil
}

func pdu(name string, v any) gosnmp.SnmpPDU { return gosnmp.SnmpPDU{Name: name, Value: v} }

func mac(s string) []byte { hw, _ := net.ParseMAC(s); return hw }

// A 3-port switch: port 1 has one host, port 2 is an uplink with an LLDP
// neighbor, port 3 has a dumb hub with six hosts behind it.
func fakeSwitch() fakeWalker {
	swMAC := "00:1b:54:00:00:01"
	w := fakeWalker{
		pdu(oidSystem+".1.0", []byte("Cisco IOS Software, C2960")),
		pdu(oidSystem+".5.0", []byte("access-sw-3")),
		pdu(oidSystem+".6.0", []byte("Building A, room 101")),
		pdu(oidIfPhysAddress+".10101", mac(swMAC)),
		pdu(oidIfPhysAddress+".10102", mac(swMAC)),
		pdu(oidIfName+".10101", []byte("Gi0/1")),
		pdu(oidIfName+".10102", []byte("Gi0/2")),
		pdu(oidIfName+".10103", []byte("Gi0/3")),
		pdu(oidBasePortIfIdx+".1", 10101),
		pdu(oidBasePortIfIdx+".2", 10102),
		pdu(oidBasePortIfIdx+".3", 10103),
		pdu(oidLLDPLocal+".1.0", 4),
		pdu(oidLLDPLocal+".2.0", mac(swMAC)),
		pdu(oidLLDPLocal+".7.1.4.2", []byte("Gi0/2")),
		// LLDP neighbor on local port 2: core switch, its port Te1/1.
		pdu(oidLLDPRem+".4.0.2.1", 4),
		pdu(oidLLDPRem+".5.0.2.1", mac("00:1b:54:99:99:99")),
		pdu(oidLLDPRem+".6.0.2.1", 5),
		pdu(oidLLDPRem+".7.0.2.1", []byte("Te1/1")),
		pdu(oidLLDPRem+".9.0.2.1", []byte("core-sw")),
		// The switch's own MAC in the FDB with status self: must be skipped.
		pdu(oidDot1dFdbPort+"."+macOIDs(swMAC), 0),
		pdu(oidDot1dFdbState+"."+macOIDs(swMAC), fdbStatusSelf),
		// Host on port 1.
		pdu(oidDot1dFdbPort+"."+macOIDs("00:11:32:aa:aa:aa"), 1),
		// Core switch MAC learned on the uplink.
		pdu(oidDot1dFdbPort+"."+macOIDs("00:1b:54:99:99:99"), 2),
		// ARP: the host's IP.
		pdu(oidIPNetToMedia+".10101.192.168.1.50", mac("00:11:32:aa:aa:aa")),
	}
	for i := range 6 {
		m := net.HardwareAddr{0x00, 0x11, 0x32, 0xbb, 0xbb, byte(i)}
		w = append(w, pdu(oidDot1dFdbPort+"."+macOIDs(m.String()), 3))
	}
	return w
}

func macOIDs(m string) string {
	hw, _ := net.ParseMAC(m)
	var sb strings.Builder
	for i, b := range hw {
		if i > 0 {
			sb.WriteByte('.')
		}
		sb.WriteString(strconvItoa(int(b)))
	}
	return sb.String()
}

func strconvItoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func TestSNMPTopology(t *testing.T) {
	s, n := testSession()
	st, err := pollDevice(s, n, netip.MustParseAddr("192.168.1.2"), fakeSwitch(), 4)
	if err != nil {
		t.Fatal(err)
	}
	if st.name != "access-sw-3" || st.fdb != 8 || st.lldp != 1 || st.arp != 1 {
		t.Fatalf("stats %+v", st)
	}
	g := resolve.Build(s.Flush())

	byLabel := map[string]*resolve.Device{}
	for _, d := range g.Devices {
		byLabel[d.Label] = d
	}
	sw, core := byLabel["access-sw-3"], byLabel["core-sw"]
	if sw == nil || core == nil {
		t.Fatalf("missing switches: %v", byLabel)
	}
	if sw.Best("location.text") != "Building A, room 101" || sw.Best("role") != "switch" {
		t.Fatalf("switch claims %+v", sw.Claims)
	}
	host := byLabel["192.168.1.50"] // MAC from FDB + IP from ARP: one device
	if host == nil {
		t.Fatalf("host not merged from FDB+ARP: %v", byLabel)
	}
	kinds := map[string]int{}
	for _, l := range g.Links {
		kinds[l.Kind]++
		switch {
		case l.A == host.ID && l.B == sw.ID:
			if l.Kind != "l2" || l.Detail != "? ↔ Gi0/1" {
				t.Errorf("host link %+v", l)
			}
		case l.A == core.ID && l.B == sw.ID && l.Kind == "l2":
			if l.Detail != "Te1/1 ↔ Gi0/2" {
				t.Errorf("lldp link %+v", l)
			}
		}
	}
	// 1 host link + LLDP link (+ the core MAC seen on the uplink, as "behind")
	// and 6 hub hosts behind port 3.
	if kinds["l2"] != 2 || kinds["l2-behind"] != 7 {
		t.Fatalf("link kinds %v", kinds)
	}
}

// A switch polled over SNMP and the same switch heard via passive LLDP on
// the wire must be one device: both use the same lldp: chassis identifier.
func TestSNMPMergesWithPassiveLLDP(t *testing.T) {
	s, n := testSession()
	if _, err := pollDevice(s, n, netip.MustParseAddr("192.168.1.2"), fakeSwitch(), 4); err != nil {
		t.Fatal(err)
	}
	active := s.Flush()

	other := NewSession(ed25519.NewKeyFromSeed(bytes.Repeat([]byte{9}, 32)), io.Discard)
	id, _ := ChassisID(4, mac("00:1b:54:00:00:01"))
	other.Emit(obs.Body{Mode: obs.ModePassive, Method: "lldp", Subject: []string{id},
		Claims: []obs.Claim{{K: "name", V: "access-sw-3"}, {K: "link.l2", V: "mac:00:11:32:cc:cc:cc"},
			{K: "link.port.a", V: "Gi0/5"}, {K: "link.port.b", V: "eth0"}}})
	other.Emit(obs.Body{Mode: obs.ModeLocal, Method: "self", Subject: []string{"rkm:bb", "mac:00:11:32:cc:cc:cc"}})

	g := resolve.Build(append(active, other.Flush()...))
	var sws []*resolve.Device
	for _, d := range g.Devices {
		if d.Label == "access-sw-3" {
			sws = append(sws, d)
		}
	}
	if len(sws) != 1 || len(sws[0].Modes) != 2 {
		t.Fatalf("want one switch seen active+passive, got %d", len(sws))
	}
}

// Two observers in different networks trace to the same destination; their
// paths share public routers, which must become the same devices.
func TestTracePathsMergeOnPublicHops(t *testing.T) {
	sA, nA := testSession()
	sB := NewSession(ed25519.NewKeyFromSeed(bytes.Repeat([]byte{8}, 32)), io.Discard)
	nB := &NetCtx{Iface: net.Interface{Name: "wlan0"}, Prefix: netip.MustParsePrefix("10.0.0.7/24"), NetID: "nother"}
	a := func(s string) netip.Addr { return netip.MustParseAddr(s) }
	emitPath(sA, nA, a("9.9.9.9"), []hop{
		{ttl: 1, addr: a("192.168.1.1")}, {ttl: 2, addr: a("100.64.3.1")}, // CGNAT: rip, not global
		{ttl: 3}, {ttl: 4, addr: a("80.80.80.1")}, {ttl: 5, addr: a("9.9.9.9"), reached: true}})
	emitPath(sB, nB, a("9.9.9.9"), []hop{
		{ttl: 1, addr: a("10.0.0.1")}, {ttl: 2, addr: a("80.80.80.1")}, {ttl: 3, addr: a("9.9.9.9"), reached: true}})

	g := resolve.Build(append(sA.Flush(), sB.Flush()...))
	var shared, cgnat *resolve.Device
	for _, d := range g.Devices {
		for _, id := range d.Idents {
			switch id {
			case "gip:80.80.80.1":
				shared = d
			case "rip:ntest/100.64.3.1":
				cgnat = d
			}
		}
	}
	if shared == nil || len(shared.Observers) != 2 {
		t.Fatalf("public hop not shared: %+v", shared)
	}
	if cgnat == nil || len(cgnat.Networks) != 0 {
		t.Fatalf("CGNAT hop must be routed (no LAN membership): %+v", cgnat)
	}
	var gapLink bool
	for _, l := range g.Links {
		if l.A == shared.ID && l.B == cgnat.ID && l.Kind == "l3" {
			gapLink = true
		}
	}
	if !gapLink {
		t.Fatalf("missing l3 link across silent hop: %+v", g.Links)
	}
}

func TestParseRecvErr(t *testing.T) {
	ee := unix.SockExtendedErr{Errno: uint32(unix.EHOSTUNREACH), Origin: unix.SO_EE_ORIGIN_ICMP, Type: 11}
	data := make([]byte, int(unsafe.Sizeof(ee))+16)
	binary.NativeEndian.PutUint32(data[0:], ee.Errno)
	data[4], data[5], data[6] = ee.Origin, ee.Type, ee.Code
	off := data[unsafe.Sizeof(ee):]
	binary.NativeEndian.PutUint16(off[0:], unix.AF_INET)
	copy(off[4:8], []byte{203, 0, 113, 9})

	from, port, kind, ok := parseRecvErr(data, 33436)
	if !ok || from != netip.MustParseAddr("203.0.113.9") || port != 33436 || kind != errTimeExceeded {
		t.Fatalf("got %v %d %v %v", from, port, kind, ok)
	}
	data[5], data[6] = 3, 3
	if _, _, kind, _ := parseRecvErr(data, 0); kind != errDestination {
		t.Fatal("port unreachable must mean destination reached")
	}
	data[6] = 1
	if _, _, kind, _ := parseRecvErr(data, 0); kind != errUnreachable {
		t.Fatal("host unreachable must end the path, not reach it")
	}
}
