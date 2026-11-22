//go:build linux

package collect

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/gopacket/gopacket"
	"github.com/gopacket/gopacket/layers"
	"github.com/mdlayher/packet"
	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

var lldpMulticast = net.HardwareAddr{0x01, 0x80, 0xc2, 0x00, 0x00, 0x0e}

// Sniff listens to raw frames and extracts identity-bearing protocols. It
// never transmits. Requires CAP_NET_RAW; without it ErrNoRaw is returned so
// the caller can continue with the unprivileged listeners.
type Sniff struct{}

var ErrNoRaw = errors.New("raw sockets not permitted (needs root or CAP_NET_RAW)")

func (Sniff) Name() string { return "sniff" }

func (Sniff) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	var conns []*packet.Conn
	for i := range nets {
		c, err := packet.Listen(&nets[i].Iface, packet.Raw, unix.ETH_P_ALL, nil)
		if err != nil {
			for _, c := range conns {
				c.Close()
			}
			if errors.Is(err, os.ErrPermission) || errors.Is(err, unix.EPERM) {
				return ErrNoRaw
			}
			return err
		}
		c.JoinGroup(lldpMulticast) // LLDP frames are not delivered otherwise
		conns = append(conns, c)
	}
	go func() {
		<-ctx.Done()
		for _, c := range conns {
			c.Close()
		}
	}()
	var wg sync.WaitGroup
	for i, c := range conns {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sniffLoop(s, &nets[i], c)
		}()
	}
	wg.Wait()
	return ctx.Err()
}

func sniffLoop(s *Session, n *NetCtx, c *packet.Conn) {
	buf := make([]byte, 65536)
	for {
		nr, _, err := c.ReadFrom(buf)
		if err != nil {
			return
		}
		handleFrame(s, n, buf[:nr])
	}
}

func handleFrame(s *Session, n *NetCtx, frame []byte) {
	p := gopacket.NewPacket(frame, layers.LayerTypeEthernet, gopacket.DecodeOptions{Lazy: true, NoCopy: true})
	eth, _ := p.Layer(layers.LayerTypeEthernet).(*layers.Ethernet)
	if eth == nil {
		return
	}
	emit := func(method string, subject []string, claims ...obs.Claim) {
		s.Emit(obs.Body{Mode: obs.ModePassive, Method: method, Vantage: n.Vantage(), Subject: subject, Claims: claims})
	}
	switch {
	case p.Layer(layers.LayerTypeARP) != nil:
		a := p.Layer(layers.LayerTypeARP).(*layers.ARP)
		ip, _ := netip.AddrFromSlice(a.SourceProtAddress)
		if ip.IsUnspecified() || !n.Contains(ip) { // ARP probes (0.0.0.0) carry no binding
			return
		}
		emit("arp", obs.Idents(n.MACID(a.SourceHwAddress), n.IPID(ip)), claim("seen.l2", "arp"))

	case p.Layer(layers.LayerTypeDHCPv4) != nil:
		handleDHCP(n, p.Layer(layers.LayerTypeDHCPv4).(*layers.DHCPv4), emit)

	case p.Layer(layers.LayerTypeICMPv6NeighborAdvertisement) != nil,
		p.Layer(layers.LayerTypeICMPv6NeighborSolicitation) != nil,
		p.Layer(layers.LayerTypeICMPv6RouterAdvertisement) != nil:
		ip6, _ := p.Layer(layers.LayerTypeIPv6).(*layers.IPv6)
		if ip6 == nil {
			return
		}
		src, _ := netip.AddrFromSlice(ip6.SrcIP)
		if src.IsUnspecified() { // DAD probe
			return
		}
		var claims []obs.Claim
		if p.Layer(layers.LayerTypeICMPv6RouterAdvertisement) != nil {
			claims = append(claims, claim("role", "router"))
		}
		emit("ndp", obs.Idents(n.MACID(eth.SrcMAC), n.IPID(src)), append(claims, claim("seen.l2", "ndp"))...)

	case p.Layer(layers.LayerTypeLinkLayerDiscovery) != nil:
		handleLLDP(n, p, emit)
	}
}

type emitFn func(method string, subject []string, claims ...obs.Claim)

func handleDHCP(n *NetCtx, d *layers.DHCPv4, emit emitFn) {
	var msgType layers.DHCPMsgType
	var hostname, vendor string
	var fp []string
	for _, o := range d.Options {
		switch o.Type {
		case layers.DHCPOptMessageType:
			if len(o.Data) == 1 {
				msgType = layers.DHCPMsgType(o.Data[0])
			}
		case layers.DHCPOptHostname:
			hostname = string(o.Data)
		case layers.DHCPOptClassID:
			vendor = string(o.Data)
		case layers.DHCPOptParamsRequest:
			for _, b := range o.Data {
				fp = append(fp, strconv.Itoa(int(b)))
			}
		}
	}
	client := n.MACID(d.ClientHWAddr)
	switch msgType {
	case layers.DHCPMsgTypeDiscover, layers.DHCPMsgTypeRequest, layers.DHCPMsgTypeInform:
		claims := []obs.Claim{claim("dhcp.vendor", vendor), claim("hostname", hostname)}
		if len(fp) > 0 {
			claims = append(claims, claim("dhcp.fp", strings.Join(fp, ",")))
		}
		emit("dhcp", obs.Idents(client, obs.NameID("host", n.NetID, hostname)), claims...)
	case layers.DHCPMsgTypeAck:
		ip, _ := netip.AddrFromSlice(d.YourClientIP.To4())
		emit("dhcp", obs.Idents(client, n.IPID(ip)), claim("dhcp.lease", "ack"))
		srv, _ := netip.AddrFromSlice(d.NextServerIP.To4())
		for _, o := range d.Options {
			if o.Type == layers.DHCPOptServerID && len(o.Data) == 4 {
				srv = netip.AddrFrom4([4]byte(o.Data))
			}
		}
		if srv.IsValid() && !srv.IsUnspecified() {
			emit("dhcp", []string{n.IPID(srv)}, claim("role", "dhcp-server"))
		}
	}
}

func handleLLDP(n *NetCtx, p gopacket.Packet, emit emitFn) {
	l := p.Layer(layers.LayerTypeLinkLayerDiscovery).(*layers.LinkLayerDiscovery)
	chassis, mac := ChassisID(int(l.ChassisID.Subtype), l.ChassisID.ID)
	port := string(l.PortID.ID)
	if l.PortID.Subtype == layers.LLDPPortIDSubtypeMACAddr {
		port = net.HardwareAddr(l.PortID.ID).String()
	}
	claims := []obs.Claim{claim("lldp.port", port), claim("role", "switch")}
	if info, ok := p.Layer(layers.LayerTypeLinkLayerDiscoveryInfo).(*layers.LinkLayerDiscoveryInfo); ok {
		claims = append(claims, claim("name", info.SysName), claim("os.desc", info.SysDescription),
			claim("lldp.port.desc", info.PortDescription))
		if ip, ok := netip.AddrFromSlice(info.MgmtAddress.Address); ok && len(info.MgmtAddress.Address) >= 4 {
			claims = append(claims, claim("mgmt.addr", ip.String()))
		}
	}
	// The neighbor is linked to *our* interface on this network: its port on
	// its side (a), our interface name on ours (b).
	claims = append(claims, claim("link.l2", n.MACID(n.Iface.HardwareAddr)),
		claim("link.port.a", port), claim("link.port.b", n.Iface.Name))
	emit("lldp", obs.Idents(chassis, obs.MACID(mac, n.NetID)), claims...)
}
