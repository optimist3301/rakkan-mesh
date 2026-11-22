//go:build linux

package collect

import (
	"bufio"
	"bytes"
	"context"
	"encoding/xml"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

var ssdpGroup = netip.MustParseAddr("239.255.255.250")

type ssdpMsg struct {
	method  string // "NOTIFY", "M-SEARCH" or "" for responses
	headers map[string]string
}

func parseSSDP(b []byte) (*ssdpMsg, bool) {
	r := bufio.NewReader(bytes.NewReader(b))
	line, err := r.ReadString('\n')
	if err != nil {
		return nil, false
	}
	m := &ssdpMsg{headers: map[string]string{}}
	switch {
	case strings.HasPrefix(line, "NOTIFY"):
		m.method = "NOTIFY"
	case strings.HasPrefix(line, "M-SEARCH"):
		m.method = "M-SEARCH"
	case strings.HasPrefix(line, "HTTP/"):
	default:
		return nil, false
	}
	for {
		l, err := r.ReadString('\n')
		k, v, ok := strings.Cut(strings.TrimSpace(l), ":")
		if ok {
			m.headers[strings.ToUpper(strings.TrimSpace(k))] = strings.TrimSpace(v)
		}
		if err != nil {
			break
		}
	}
	return m, true
}

// uuidFromUSN extracts "uuid:xxxx" from "uuid:xxxx::urn:schemas-upnp-org:...".
func uuidFromUSN(usn string) string {
	u, _, _ := strings.Cut(usn, "::")
	if !strings.HasPrefix(strings.ToLower(u), "uuid:") {
		return ""
	}
	return "uuid:" + strings.ToLower(strings.TrimSpace(u[5:]))
}

func emitSSDP(s *Session, n *NetCtx, mode obs.Mode, src netip.Addr, m *ssdpMsg) string {
	if !n.Contains(src) {
		return ""
	}
	if m.method == "M-SEARCH" { // someone searching: tells us a client exists
		s.Emit(obs.Body{Mode: mode, Method: "ssdp", Vantage: n.Vantage(), Subject: []string{n.IPID(src)},
			Claims: []obs.Claim{claim("upnp.client", m.headers["USER-AGENT"])}})
		return ""
	}
	if m.method == "NOTIFY" && m.headers["NTS"] == "ssdp:byebye" {
		return ""
	}
	typ := m.headers["ST"]
	if typ == "" {
		typ = m.headers["NT"]
	}
	claims := []obs.Claim{claim("upnp.server", m.headers["SERVER"])}
	if strings.HasPrefix(typ, "urn:") {
		claims = append(claims, claim("upnp.type", typ))
	}
	s.Emit(obs.Body{Mode: mode, Method: "ssdp", Vantage: n.Vantage(),
		Subject: obs.Idents(n.IPID(src), uuidFromUSN(m.headers["USN"])), Claims: claims})
	return m.headers["LOCATION"]
}

// SSDPSearch sends an M-SEARCH and fetches the device descriptions.
type SSDPSearch struct{ Wait time.Duration }

func (SSDPSearch) Name() string { return "ssdp" }

func (c SSDPSearch) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	for i := range nets {
		n := &nets[i]
		conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: n.Prefix.Addr().AsSlice()})
		if err != nil {
			return err
		}
		req := "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\nST: ssdp:all\r\n\r\n"
		dst := &net.UDPAddr{IP: ssdpGroup.AsSlice(), Port: 1900}
		for range 2 {
			conn.WriteTo([]byte(req), dst)
		}
		locations := map[string]netip.Addr{}
		deadline := time.Now().Add(c.Wait)
		buf := make([]byte, 4096)
		for ctx.Err() == nil {
			conn.SetReadDeadline(deadline)
			nr, src, err := conn.ReadFromUDP(buf)
			if err != nil {
				break
			}
			if m, ok := parseSSDP(buf[:nr]); ok {
				if loc := emitSSDP(s, n, obs.ModeActive, addrFromIP(src.IP), m); loc != "" {
					locations[loc] = addrFromIP(src.IP)
				}
			}
		}
		conn.Close()
		var wg sync.WaitGroup
		for loc, src := range locations {
			wg.Add(1)
			go func() { defer wg.Done(); fetchDescription(ctx, s, n, loc, src) }()
		}
		wg.Wait()
	}
	return nil
}

type upnpRoot struct {
	Device struct {
		FriendlyName string `xml:"friendlyName"`
		Manufacturer string `xml:"manufacturer"`
		ModelName    string `xml:"modelName"`
		ModelNumber  string `xml:"modelNumber"`
		Serial       string `xml:"serialNumber"`
		UDN          string `xml:"UDN"`
		DeviceType   string `xml:"deviceType"`
	} `xml:"device"`
}

var httpClient = &http.Client{Timeout: 3 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

// fetchDescription reads the UPnP device XML, only from the announcing host.
func fetchDescription(ctx context.Context, s *Session, n *NetCtx, loc string, src netip.Addr) {
	u, err := url.Parse(loc)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return
	}
	if h, err := netip.ParseAddr(u.Hostname()); err != nil || h != src {
		return // don't follow descriptions pointing elsewhere
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, loc, nil)
	resp, err := httpClient.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	var root upnpRoot
	if err := xml.NewDecoder(io.LimitReader(resp.Body, 256<<10)).Decode(&root); err != nil {
		return
	}
	d := root.Device
	model := strings.TrimSpace(d.ModelName + " " + d.ModelNumber)
	s.Emit(obs.Body{Mode: obs.ModeActive, Method: "upnp-desc", Vantage: n.Vantage(),
		Subject: obs.Idents(n.IPID(src), uuidFromUSN(d.UDN)),
		Claims: []obs.Claim{claim("name", d.FriendlyName), claim("vendor", d.Manufacturer),
			claim("model", model), claim("serial", d.Serial), claim("upnp.device", d.DeviceType)}})
}

// SSDPListen passively records NOTIFY announcements and other hosts' searches.
type SSDPListen struct{}

func (SSDPListen) Name() string { return "ssdp-listen" }

func (SSDPListen) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	pc, err := listenGroup(ctx, ssdpGroup, 1900, nets)
	if err != nil {
		return err
	}
	buf := make([]byte, 4096)
	for {
		nr, cm, src, err := pc.ReadFrom(buf)
		if err != nil {
			return ctx.Err()
		}
		ua, ok := src.(*net.UDPAddr)
		if cm == nil || !ok {
			continue
		}
		if n := netByIndex(nets, cm.IfIndex); n != nil {
			if m, ok := parseSSDP(buf[:nr]); ok {
				emitSSDP(s, n, obs.ModePassive, addrFromIP(ua.IP), m)
			}
		}
	}
}
