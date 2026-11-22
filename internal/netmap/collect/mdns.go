//go:build linux

package collect

import (
	"context"
	"net"
	"net/netip"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/net/dns/dnsmessage"
	"golang.org/x/net/ipv4"
	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

var mdnsGroup = netip.MustParseAddr("224.0.0.251")

// commonServices are queried directly as well, since not every responder
// answers the _services._dns-sd._udp meta-query.
var commonServices = []string{
	"_workstation._tcp", "_device-info._tcp", "_http._tcp", "_ipp._tcp", "_ipps._tcp",
	"_printer._tcp", "_pdl-datastream._tcp", "_scanner._tcp", "_uscan._tcp", "_airplay._tcp", "_raop._tcp",
	"_googlecast._tcp", "_spotify-connect._tcp", "_ssh._tcp", "_sftp-ssh._tcp", "_smb._tcp",
	"_afpovertcp._tcp", "_companion-link._tcp", "_hap._tcp", "_homekit._tcp", "_sonos._tcp",
	"_androidtvremote2._tcp", "_matter._tcp", "_esphomelib._tcp", "_mqtt._tcp", "_rdlink._tcp",
}

var txtModelKeys = []string{"model", "md", "ty", "usb_MDL", "product", "am", "rpMd"}
var txtVendorKeys = []string{"manufacturer", "usb_MFG", "mfg", "vendor"}

func stripLocal(name string) string { return strings.TrimSuffix(name, ".") }

// serviceOf splits "My Printer._ipp._tcp.local." into ("My Printer", "_ipp._tcp").
func serviceOf(name string) (instance, svc string) {
	name = stripLocal(name)
	i := strings.Index(name, "._")
	if i < 0 {
		return "", ""
	}
	instance, rest := name[:i], name[i+1:]
	rest = strings.TrimSuffix(rest, ".local")
	return strings.ReplaceAll(instance, `\`, ""), rest
}

// parseMDNS turns one mDNS response into observations. Returns the service
// types announced via the meta-query (for follow-up queries).
func parseMDNS(s *Session, n *NetCtx, mode obs.Mode, pkt []byte) []string {
	var p dnsmessage.Parser
	h, err := p.Start(pkt)
	if err != nil || !h.Response {
		return nil
	}
	if err := p.SkipAllQuestions(); err != nil {
		return nil
	}
	addrs := map[string][]string{}
	type srvRec struct {
		target string
		port   uint16
	}
	srvs := map[string]srvRec{}
	txts := map[string][]string{}
	var types []string
	onSRV := func(inst, target string, port uint16) { srvs[inst] = srvRec{target, port} }
	sections := []struct {
		next func() (dnsmessage.ResourceHeader, error)
		skip func() error
	}{{p.AnswerHeader, p.SkipAnswer}, {p.AuthorityHeader, p.SkipAuthority}, {p.AdditionalHeader, p.SkipAdditional}}
sections:
	for _, sec := range sections {
		for {
			rh, err := sec.next()
			if err == dnsmessage.ErrSectionDone { // the parser advances to the next section
				break
			}
			if err != nil || readResource(&p, rh, sec.skip, addrs, onSRV, txts, &types) != nil {
				break sections // malformed: keep what was parsed so far
			}
		}
	}

	now := time.Now().UnixMilli()
	emit := func(subject []string, claims ...obs.Claim) {
		s.Emit(obs.Body{Time: now, Mode: mode, Method: "mdns", Vantage: n.Vantage(), Subject: subject, Claims: claims})
	}
	for host, list := range addrs {
		subj := []string{obs.NameID("mdns", n.NetID, host)}
		for _, a := range list {
			if ip, err := netip.ParseAddr(a); err == nil && n.Contains(ip) {
				subj = append(subj, n.IPID(ip))
			}
		}
		emit(subj, claim("mdns.name", stripLocal(host)))
	}
	for inst, srv := range srvs {
		label, svc := serviceOf(inst)
		target := obs.NameID("mdns", n.NetID, srv.target)
		claims := []obs.Claim{claim("service", "mdns/"+svc+":"+strconv.Itoa(int(srv.port)))}
		if label != "" {
			claims = append(claims, claim("mdns.instance", label))
		}
		claims = append(claims, txtClaims(txts[inst])...)
		emit([]string{target}, claims...)
	}
	for inst, kv := range txts {
		label, svc := serviceOf(inst)
		if svc == "_device-info._tcp" && label != "" {
			emit([]string{obs.NameID("mdns", n.NetID, label+".local")}, txtClaims(kv)...)
		}
	}
	return types
}

func txtClaims(kv []string) []obs.Claim {
	var out []obs.Claim
	get := func(keys []string) string {
		for _, k := range keys {
			for _, e := range kv {
				if key, v, ok := strings.Cut(e, "="); ok && strings.EqualFold(key, k) && meaningful(v) {
					return v
				}
			}
		}
		return ""
	}
	if m := get(txtModelKeys); m != "" {
		out = append(out, claimC("model", m, 900))
	}
	if v := get(txtVendorKeys); v != "" {
		out = append(out, claimC("vendor", v, 900))
	}
	if v := get([]string{"fn", "n"}); v != "" {
		out = append(out, claimC("name", v, 800))
	}
	return out
}

// meaningful rejects flag-like TXT values ("0,1,2" is RAOP's md= metadata
// list, not a model).
func meaningful(v string) bool {
	return strings.Trim(v, "0123456789,.") != ""
}

// readResource consumes one resource body. Any error must abort the packet:
// after a failed body parse dnsmessage re-returns the same header forever.
func readResource(p *dnsmessage.Parser, rh dnsmessage.ResourceHeader, skip func() error, addrs map[string][]string,
	srv func(inst, target string, port uint16), txts map[string][]string, types *[]string) error {
	name := rh.Name.String()
	switch rh.Type {
	case dnsmessage.TypeA:
		r, err := p.AResource()
		if err != nil {
			return err
		}
		addrs[name] = append(addrs[name], netip.AddrFrom4(r.A).String())
	case dnsmessage.TypeAAAA:
		r, err := p.AAAAResource()
		if err != nil {
			return err
		}
		addrs[name] = append(addrs[name], netip.AddrFrom16(r.AAAA).String())
	case dnsmessage.TypeSRV:
		r, err := p.SRVResource()
		if err != nil {
			return err
		}
		srv(name, r.Target.String(), r.Port)
	case dnsmessage.TypeTXT:
		r, err := p.TXTResource()
		if err != nil {
			return err
		}
		txts[name] = append(txts[name], r.TXT...)
	case dnsmessage.TypePTR:
		r, err := p.PTRResource()
		if err != nil {
			return err
		}
		if strings.HasPrefix(name, "_services._dns-sd._udp.") {
			*types = append(*types, strings.TrimSuffix(stripLocal(r.PTR.String()), ".local"))
		}
	default:
		return skip()
	}
	return nil
}

func mdnsQuery(names []string) []byte {
	b := dnsmessage.NewBuilder(nil, dnsmessage.Header{})
	b.StartQuestions()
	for _, n := range names {
		q, err := dnsmessage.NewName(n + ".local.")
		if err != nil {
			continue
		}
		b.Question(dnsmessage.Question{Name: q, Type: dnsmessage.TypePTR, Class: dnsmessage.ClassINET})
	}
	out, _ := b.Finish()
	return out
}

// MDNSQuery actively asks for services (legacy unicast query: responders
// reply straight to our port, so no special privileges are needed).
type MDNSQuery struct{ Wait time.Duration }

func (MDNSQuery) Name() string { return "mdns" }

func (c MDNSQuery) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	for i := range nets {
		n := &nets[i]
		conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: n.Prefix.Addr().AsSlice()})
		if err != nil {
			return err
		}
		dst := &net.UDPAddr{IP: mdnsGroup.AsSlice(), Port: 5353}
		found := map[string]bool{}
		round := func(names []string) {
			for start := 0; start < len(names); start += 8 {
				conn.WriteTo(mdnsQuery(names[start:min(start+8, len(names))]), dst)
			}
			deadline := time.Now().Add(c.Wait)
			buf := make([]byte, 9000)
			for ctx.Err() == nil {
				conn.SetReadDeadline(deadline)
				nr, _, err := conn.ReadFromUDP(buf)
				if err != nil {
					return
				}
				for _, t := range parseMDNS(s, n, obs.ModeActive, buf[:nr]) {
					found[t] = true
				}
			}
		}
		round([]string{"_services._dns-sd._udp"})
		names := append([]string{}, commonServices...)
		for t := range found {
			names = append(names, t)
		}
		round(names)
		conn.Close()
	}
	return nil
}

// listenGroup binds a shared multicast port (SO_REUSEADDR/PORT, so it
// coexists with avahi or systemd-resolved) and joins the group on each net.
func listenGroup(ctx context.Context, group netip.Addr, port int, nets []NetCtx) (*ipv4.PacketConn, error) {
	lc := net.ListenConfig{Control: func(_, _ string, rc syscall.RawConn) error {
		var serr error
		rc.Control(func(fd uintptr) {
			serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEADDR, 1)
			if serr == nil {
				serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
			}
		})
		return serr
	}}
	c, err := lc.ListenPacket(ctx, "udp4", "0.0.0.0:"+strconv.Itoa(port))
	if err != nil {
		return nil, err
	}
	pc := ipv4.NewPacketConn(c)
	for i := range nets {
		if err := pc.JoinGroup(&nets[i].Iface, &net.UDPAddr{IP: group.AsSlice()}); err != nil {
			c.Close()
			return nil, err
		}
	}
	pc.SetControlMessage(ipv4.FlagInterface, true)
	go func() { <-ctx.Done(); c.Close() }()
	return pc, nil
}

func netByIndex(nets []NetCtx, idx int) *NetCtx {
	for i := range nets {
		if nets[i].Iface.Index == idx {
			return &nets[i]
		}
	}
	return nil
}

// MDNSListen passively records mDNS announcements and responses others ask for.
type MDNSListen struct{}

func (MDNSListen) Name() string { return "mdns-listen" }

func (MDNSListen) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	pc, err := listenGroup(ctx, mdnsGroup, 5353, nets)
	if err != nil {
		return err
	}
	buf := make([]byte, 9000)
	for {
		nr, cm, _, err := pc.ReadFrom(buf)
		if err != nil {
			return ctx.Err()
		}
		if cm == nil {
			continue
		}
		if n := netByIndex(nets, cm.IfIndex); n != nil {
			parseMDNS(s, n, obs.ModePassive, buf[:nr])
		}
	}
}
