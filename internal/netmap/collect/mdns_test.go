//go:build linux

package collect

import (
	"bytes"
	"crypto/ed25519"
	"io"
	"net"
	"net/netip"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

func testSession() (*Session, *NetCtx) {
	s := NewSession(ed25519.NewKeyFromSeed(bytes.Repeat([]byte{7}, 32)), io.Discard)
	n := &NetCtx{Iface: net.Interface{Name: "eth0"}, Prefix: netip.MustParsePrefix("192.168.1.10/24"), NetID: "ntest"}
	return s, n
}

func sampleResponse(t testing.TB) []byte {
	b := dnsmessage.NewBuilder(nil, dnsmessage.Header{Response: true, Authoritative: true})
	b.EnableCompression()
	b.StartAnswers()
	inst := dnsmessage.MustNewName("Office Printer._ipp._tcp.local.")
	host := dnsmessage.MustNewName("printer.local.")
	hdr := func(n dnsmessage.Name) dnsmessage.ResourceHeader {
		return dnsmessage.ResourceHeader{Name: n, Class: dnsmessage.ClassINET, TTL: 120}
	}
	b.PTRResource(hdr(dnsmessage.MustNewName("_ipp._tcp.local.")), dnsmessage.PTRResource{PTR: inst})
	b.SRVResource(hdr(inst), dnsmessage.SRVResource{Target: host, Port: 631})
	b.TXTResource(hdr(inst), dnsmessage.TXTResource{TXT: []string{"ty=HP LaserJet 400", "usb_MFG=HP", "md=0,1,2"}})
	b.AResource(hdr(host), dnsmessage.AResource{A: [4]byte{192, 168, 1, 50}})
	out, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestParseMDNS(t *testing.T) {
	s, n := testSession()
	parseMDNS(s, n, obs.ModePassive, sampleResponse(t))
	list := s.Flush()
	var bound, svc bool
	for _, o := range list {
		if len(o.Subject) == 2 && o.Subject[0] == "ip:ntest/192.168.1.50" && o.Subject[1] == "mdns:ntest/printer.local" {
			bound = true
		}
		if v, _ := o.Get("model"); v == "HP LaserJet 400" {
			if vend, _ := o.Get("vendor"); vend != "HP" {
				t.Errorf("vendor = %q", vend)
			}
			svc = true
		}
	}
	if !bound || !svc {
		for _, o := range list {
			t.Logf("%v %v", o.Subject, o.Claims)
		}
		t.Fatalf("bound=%v service=%v", bound, svc)
	}
}

// Truncated or corrupted packets from the air must never hang the parser.
func TestParseMDNSMalformedTerminates(t *testing.T) {
	pkt := sampleResponse(t)
	done := make(chan struct{})
	go func() {
		defer close(done)
		s, n := testSession()
		for i := range len(pkt) {
			parseMDNS(s, n, obs.ModePassive, pkt[:i])
			bad := bytes.Clone(pkt)
			bad[i] ^= 0xff
			parseMDNS(s, n, obs.ModePassive, bad)
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("parseMDNS hangs on malformed input")
	}
}

func FuzzParseMDNS(f *testing.F) {
	f.Add(sampleResponse(f))
	s, n := testSession()
	f.Fuzz(func(t *testing.T, pkt []byte) {
		parseMDNS(s, n, obs.ModePassive, pkt)
		s.Flush()
	})
}
