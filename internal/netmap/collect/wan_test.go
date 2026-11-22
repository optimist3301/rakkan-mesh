//go:build linux

package collect

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

func udpServer(t *testing.T, reply func(req []byte) []byte) netip.AddrPort {
	t.Helper()
	pc, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { pc.Close() })
	go func() {
		buf := make([]byte, 1500)
		for {
			n, from, err := pc.ReadFrom(buf)
			if err != nil {
				return
			}
			if r := reply(buf[:n]); r != nil {
				pc.WriteTo(r, from)
			}
		}
	}()
	return netip.MustParseAddrPort(pc.LocalAddr().String())
}

func TestNATPMP(t *testing.T) {
	ap := udpServer(t, func(req []byte) []byte {
		if len(req) != 2 || req[0] != 0 || req[1] != 0 {
			return nil
		}
		resp := make([]byte, 12)
		resp[1] = 128
		binary.BigEndian.PutUint32(resp[4:], 12345)
		copy(resp[8:], []byte{203, 0, 113, 7})
		return resp
	})
	a, err := natPMPAt(context.Background(), ap, time.Second)
	if err != nil || a != netip.MustParseAddr("203.0.113.7") {
		t.Fatalf("%v %v", a, err)
	}
	for _, bad := range [][]byte{{0, 128, 0, 3, 0, 0, 0, 0, 1, 2, 3, 4}, {0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2}} {
		if _, err := parseNATPMP(bad); err == nil {
			t.Errorf("accepted %v", bad)
		}
	}
}

func TestSTUN(t *testing.T) {
	ap := udpServer(t, func(req []byte) []byte {
		if len(req) != 20 || binary.BigEndian.Uint16(req) != 1 {
			return nil
		}
		resp := make([]byte, 20+12)
		binary.BigEndian.PutUint16(resp[0:], 0x0101)
		binary.BigEndian.PutUint16(resp[2:], 12)
		binary.BigEndian.PutUint32(resp[4:], stunMagic)
		copy(resp[8:20], req[8:20])
		binary.BigEndian.PutUint16(resp[20:], 0x0020)
		binary.BigEndian.PutUint16(resp[22:], 8)
		resp[25] = 0x01
		ip := []byte{198, 51, 100, 23}
		var m [4]byte
		binary.BigEndian.PutUint32(m[:], stunMagic)
		for i := range ip {
			resp[28+i] = ip[i] ^ m[i]
		}
		return resp
	})
	a, err := stunMapped(context.Background(), netip.Addr{}, ap.String(), 2*time.Second)
	if err != nil || a != netip.MustParseAddr("198.51.100.23") {
		t.Fatalf("%v %v", a, err)
	}
	// A response for someone else's transaction is ignored.
	if _, err := parseSTUN(make([]byte, 20), make([]byte, 12)); err == nil {
		t.Fatal("foreign STUN response accepted")
	}
}

const igdXML = `<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0"><device>
  <deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>
  <deviceList><device><deviceType>urn:schemas-upnp-org:device:WANDevice:1</deviceType>
    <deviceList><device><deviceType>urn:schemas-upnp-org:device:WANConnectionDevice:1</deviceType>
      <serviceList><service>
        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>
        <controlURL>/ctl/IPConn</controlURL>
      </service></serviceList>
    </device></deviceList>
  </device></deviceList>
</device></root>`

func TestUPnPIGD(t *testing.T) {
	var gotAction string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/rootDesc.xml":
			io.WriteString(w, igdXML)
		case "/ctl/IPConn":
			gotAction = r.Header.Get("SOAPAction")
			io.WriteString(w, `<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body>
<u:GetExternalIPAddressResponse xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
<NewExternalIPAddress>100.70.1.2</NewExternalIPAddress></u:GetExternalIPAddressResponse></s:Body></s:Envelope>`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	gw := netip.MustParseAddr("127.0.0.1")
	a, err := igdExternal(context.Background(), srv.URL+"/rootDesc.xml", gw, 2*time.Second)
	if err != nil || a != netip.MustParseAddr("100.70.1.2") {
		t.Fatalf("%v %v", a, err)
	}
	if !strings.HasSuffix(gotAction, `#GetExternalIPAddress"`) {
		t.Fatalf("SOAPAction %q", gotAction)
	}
	// A description hosted anywhere but the gateway is not followed.
	if _, err := igdExternal(context.Background(), srv.URL+"/rootDesc.xml", netip.MustParseAddr("192.168.1.1"), time.Second); err == nil {
		t.Fatal("followed a description off the gateway")
	}
}

// Two homes behind the same carrier NAT and the same public address: their
// maps meet at that address, while the carrier-NAT side stays local.
func TestWANLinksMaps(t *testing.T) {
	sA, nA := testSession()
	nA.GWMAC, _ = net.ParseMAC("00:11:32:00:00:01")
	emitWAN(sA, nA, netip.MustParseAddr("100.70.1.2"), "upnp", nA.MACID(nA.GWMAC))
	emitWAN(sA, nA, netip.MustParseAddr("81.2.69.160"), "stun", nA.AddrID(netip.MustParseAddr("100.70.1.2")))

	sB, nB := testSession()
	nB.NetID = "nother"
	nB.GWMAC, _ = net.ParseMAC("00:11:32:00:00:02")
	emitWAN(sB, nB, netip.MustParseAddr("81.2.69.160"), "nat-pmp", nB.MACID(nB.GWMAC))

	sA.Emit(obs.Body{Mode: obs.ModeLocal, Method: "route", Subject: []string{nA.MACID(nA.GWMAC)}, Claims: []obs.Claim{{K: "role", V: "gateway"}}})
	sB.Emit(obs.Body{Mode: obs.ModeLocal, Method: "route", Subject: []string{nB.MACID(nB.GWMAC)}, Claims: []obs.Claim{{K: "role", V: "gateway"}}})
	g := resolve.Build(append(sA.Flush(), sB.Flush()...))

	var public, cgnat *resolve.Device
	for _, d := range g.Devices {
		for _, id := range d.Idents {
			switch id {
			case "gip:81.2.69.160":
				public = d
			case "rip:ntest/100.70.1.2":
				cgnat = d
			}
		}
	}
	if public == nil || cgnat == nil || public.Best("wan.nat") != "" || cgnat.Best("wan.nat") == "" {
		t.Fatalf("public=%v cgnat=%v", public, cgnat)
	}
	wan := 0
	for _, l := range g.Links {
		if l.Kind == "wan" {
			wan++
		}
	}
	// gwA–cgnat, cgnat–public, gwB–public
	if wan != 3 {
		t.Fatalf("want 3 wan links, got %d: %+v", wan, g.Links)
	}
	for _, n := range g.Networks {
		if n.Best("wan.nat-pmp") == "" && n.Best("wan.upnp") == "" {
			t.Fatalf("network %s lacks its external address: %+v", n.ID, n.Claims)
		}
	}
}
