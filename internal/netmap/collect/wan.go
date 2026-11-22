//go:build linux

package collect

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// WAN learns how each attached network reaches the internet: the address
// its gateway has on the outside. It only asks "what is my external
// address" and never opens ports:
//
//   - NAT-PMP (RFC 6886) external-address request to the default gateway
//   - UPnP IGD GetExternalIPAddress on the gateway's own description
//   - optionally STUN (RFC 5389) to a server the user names: the address as
//     seen from the internet, which differs when the ISP adds its own NAT
//
// Results are separate observations (method "wan") so that exports can
// withhold them as a whole (--wan drop).
type WAN struct {
	STUN    string // host:port; empty = do not contact anything beyond the gateway
	Timeout time.Duration
}

func (WAN) Name() string { return "wan" }

func (c WAN) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	for i := range nets {
		n := &nets[i]
		if !n.GW.IsValid() || !n.GW.Is4() {
			continue
		}
		var wan netip.Addr
		var source string
		if a, err := natPMPExternal(ctx, n.GW, c.Timeout); err == nil {
			wan, source = a, "nat-pmp"
		} else if a, err := upnpExternal(ctx, n, c.Timeout); err == nil {
			wan, source = a, "upnp"
		} else {
			s.Log.Printf("wan %s: gateway does not tell its external address (NAT-PMP, UPnP)", n.Iface.Name)
		}
		gw := n.MACID(n.GWMAC)
		if wan.IsValid() {
			emitWAN(s, n, wan, source, gw)
		}
		if c.STUN != "" {
			pub, err := stunMapped(ctx, n.Prefix.Addr(), c.STUN, c.Timeout)
			if err != nil {
				s.Log.Printf("wan %s: stun %s: %v", n.Iface.Name, c.STUN, err)
				continue
			}
			via := gw
			if wan.IsValid() && wan != pub {
				via = n.AddrID(wan) // carrier NAT: our router's outside is not the internet yet
			}
			emitWAN(s, n, pub, "stun", via)
		}
	}
	return nil
}

func emitWAN(s *Session, n *NetCtx, addr netip.Addr, source, via string) {
	id := n.AddrID(addr) // gip: for public addresses, rip: for carrier/private ones
	claims := []obs.Claim{claim("role", "wan-address"), claim("wan.source", source), claim("link.wan", via)}
	if !obs.IsPublic(addr) {
		claims = append(claims, claim("wan.nat", "carrier-or-private"))
	}
	s.Emit(obs.Body{Mode: obs.ModeActive, Method: "wan", Vantage: n.Vantage(), Subject: []string{id}, Claims: claims})
	s.Emit(obs.Body{Mode: obs.ModeActive, Method: "wan", Vantage: n.Vantage(),
		Subject: []string{obs.NetEntityID(n.NetID)}, Claims: []obs.Claim{claim("wan."+source, addr.String())}})
}

// ---------- NAT-PMP ----------

// natPMPExternal sends the 2-byte external address request (version 0,
// opcode 0) to the gateway's port 5351.
func natPMPExternal(ctx context.Context, gw netip.Addr, timeout time.Duration) (netip.Addr, error) {
	return natPMPAt(ctx, netip.AddrPortFrom(gw, 5351), timeout)
}

func natPMPAt(ctx context.Context, ap netip.AddrPort, timeout time.Duration) (netip.Addr, error) {
	d := net.Dialer{Timeout: timeout}
	conn, err := d.DialContext(ctx, "udp4", ap.String())
	if err != nil {
		return netip.Addr{}, err
	}
	defer conn.Close()
	buf := make([]byte, 64)
	// RFC 6886 §3.1: retransmit starting at 250 ms, doubling.
	for wait := 250 * time.Millisecond; wait <= timeout; wait *= 2 {
		if _, err := conn.Write([]byte{0, 0}); err != nil {
			return netip.Addr{}, err
		}
		conn.SetReadDeadline(time.Now().Add(wait))
		n, err := conn.Read(buf)
		if err != nil {
			continue
		}
		return parseNATPMP(buf[:n])
	}
	return netip.Addr{}, errors.New("no NAT-PMP answer")
}

func parseNATPMP(b []byte) (netip.Addr, error) {
	if len(b) < 12 || b[0] != 0 || b[1] != 128 {
		return netip.Addr{}, errors.New("not a NAT-PMP external address response")
	}
	if code := binary.BigEndian.Uint16(b[2:4]); code != 0 {
		return netip.Addr{}, fmt.Errorf("NAT-PMP result code %d", code)
	}
	a := netip.AddrFrom4([4]byte(b[8:12]))
	if a.IsUnspecified() {
		return netip.Addr{}, errors.New("gateway has no external address yet")
	}
	return a, nil
}

// ---------- UPnP IGD ----------

var wanServices = []string{
	"urn:schemas-upnp-org:service:WANIPConnection:2",
	"urn:schemas-upnp-org:service:WANIPConnection:1",
	"urn:schemas-upnp-org:service:WANPPPConnection:1",
}

func upnpExternal(ctx context.Context, n *NetCtx, timeout time.Duration) (netip.Addr, error) {
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: n.Prefix.Addr().AsSlice()})
	if err != nil {
		return netip.Addr{}, err
	}
	defer conn.Close()
	req := "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\n" +
		"ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n"
	conn.WriteTo([]byte(req), &net.UDPAddr{IP: ssdpGroup.AsSlice(), Port: 1900})
	deadline := time.Now().Add(timeout)
	buf := make([]byte, 4096)
	for ctx.Err() == nil {
		conn.SetReadDeadline(deadline)
		nr, src, err := conn.ReadFromUDP(buf)
		if err != nil {
			break
		}
		// Only our own gateway's description is followed.
		if addrFromIP(src.IP) != n.GW {
			continue
		}
		if m, ok := parseSSDP(buf[:nr]); ok && m.headers["LOCATION"] != "" {
			return igdExternal(ctx, m.headers["LOCATION"], n.GW, timeout)
		}
	}
	return netip.Addr{}, errors.New("no UPnP gateway answered")
}

type igdDesc struct {
	URLBase string    `xml:"URLBase"`
	Devices []igdNode `xml:"device"`
}

type igdNode struct {
	Services []struct {
		Type       string `xml:"serviceType"`
		ControlURL string `xml:"controlURL"`
	} `xml:"serviceList>service"`
	Devices []igdNode `xml:"deviceList>device"`
}

func findWANService(nodes []igdNode) (typ, control string) {
	for _, d := range nodes {
		for _, want := range wanServices {
			for _, s := range d.Services {
				if s.Type == want {
					return s.Type, s.ControlURL
				}
			}
		}
		if t, c := findWANService(d.Devices); t != "" {
			return t, c
		}
	}
	return "", ""
}

// igdExternal reads the description at loc (which must be on gw) and asks
// the WAN connection service for its external address.
func igdExternal(ctx context.Context, loc string, gw netip.Addr, timeout time.Duration) (netip.Addr, error) {
	u, err := url.Parse(loc)
	if err != nil {
		return netip.Addr{}, err
	}
	if h, err := netip.ParseAddr(u.Hostname()); err != nil || h != gw {
		return netip.Addr{}, errors.New("description is not on the gateway")
	}
	hc := &http.Client{Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, loc, nil)
	resp, err := hc.Do(req)
	if err != nil {
		return netip.Addr{}, err
	}
	var desc igdDesc
	err = xml.NewDecoder(io.LimitReader(resp.Body, 256<<10)).Decode(&desc)
	resp.Body.Close()
	if err != nil {
		return netip.Addr{}, err
	}
	typ, control := findWANService(desc.Devices)
	if typ == "" {
		return netip.Addr{}, errors.New("gateway has no WAN connection service")
	}
	base := u
	if desc.URLBase != "" {
		if b, err := url.Parse(desc.URLBase); err == nil {
			base = b
		}
	}
	cu, err := base.Parse(control)
	if err != nil || cu.Hostname() != u.Hostname() {
		return netip.Addr{}, errors.New("control URL is not on the gateway")
	}
	body := `<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" ` +
		`s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>` +
		`<u:GetExternalIPAddress xmlns:u="` + typ + `"/></s:Body></s:Envelope>`
	req, _ = http.NewRequestWithContext(ctx, http.MethodPost, cu.String(), strings.NewReader(body))
	req.Header.Set("Content-Type", `text/xml; charset="utf-8"`)
	req.Header.Set("SOAPAction", `"`+typ+`#GetExternalIPAddress"`)
	resp, err = hc.Do(req)
	if err != nil {
		return netip.Addr{}, err
	}
	defer resp.Body.Close()
	var env struct {
		IP string `xml:"Body>GetExternalIPAddressResponse>NewExternalIPAddress"`
	}
	if err := xml.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&env); err != nil {
		return netip.Addr{}, err
	}
	a, err := netip.ParseAddr(strings.TrimSpace(env.IP))
	if err != nil || a.IsUnspecified() {
		return netip.Addr{}, fmt.Errorf("gateway returned %q", env.IP)
	}
	return a.Unmap(), nil
}

// ---------- STUN ----------

const stunMagic = 0x2112A442

// stunMapped sends one Binding request and returns the address the server
// saw (XOR-MAPPED-ADDRESS, falling back to MAPPED-ADDRESS).
func stunMapped(ctx context.Context, local netip.Addr, server string, timeout time.Duration) (netip.Addr, error) {
	d := net.Dialer{Timeout: timeout}
	if local.IsValid() {
		d.LocalAddr = &net.UDPAddr{IP: local.AsSlice()}
	}
	conn, err := d.DialContext(ctx, "udp4", server)
	if err != nil {
		return netip.Addr{}, err
	}
	defer conn.Close()
	req := make([]byte, 20)
	binary.BigEndian.PutUint16(req[0:], 0x0001) // Binding request
	binary.BigEndian.PutUint32(req[4:], stunMagic)
	rand.Read(req[8:20])
	buf := make([]byte, 1500)
	for wait := 500 * time.Millisecond; wait <= timeout; wait *= 2 {
		conn.Write(req)
		conn.SetReadDeadline(time.Now().Add(wait))
		n, err := conn.Read(buf)
		if err != nil {
			continue
		}
		if a, err := parseSTUN(buf[:n], req[8:20]); err == nil {
			return a, nil
		}
	}
	return netip.Addr{}, errors.New("no STUN answer")
}

func parseSTUN(b, txid []byte) (netip.Addr, error) {
	if len(b) < 20 || binary.BigEndian.Uint16(b[0:]) != 0x0101 || binary.BigEndian.Uint32(b[4:]) != stunMagic ||
		!bytes.Equal(b[8:20], txid) {
		return netip.Addr{}, errors.New("not our STUN response")
	}
	attrs := b[20:min(len(b), 20+int(binary.BigEndian.Uint16(b[2:])))]
	var mapped netip.Addr
	for len(attrs) >= 4 {
		t, l := binary.BigEndian.Uint16(attrs), int(binary.BigEndian.Uint16(attrs[2:]))
		if len(attrs) < 4+l {
			break
		}
		v := attrs[4 : 4+l]
		if l >= 8 && v[1] == 0x01 { // IPv4
			ip := [4]byte(v[4:8])
			switch t {
			case 0x0020: // XOR-MAPPED-ADDRESS
				var m [4]byte
				binary.BigEndian.PutUint32(m[:], stunMagic)
				for i := range ip {
					ip[i] ^= m[i]
				}
				return netip.AddrFrom4(ip), nil
			case 0x0001: // MAPPED-ADDRESS
				mapped = netip.AddrFrom4(ip)
			}
		}
		attrs = attrs[4+(l+3)&^3:] // attributes are padded to 4 bytes
	}
	if mapped.IsValid() {
		return mapped, nil
	}
	return netip.Addr{}, errors.New("STUN response without an address")
}
