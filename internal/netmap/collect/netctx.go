//go:build linux

package collect

import (
	"context"
	"crypto/ed25519"
	"net"
	"net/netip"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// NetCtx is one attached L2 network as seen from this host.
type NetCtx struct {
	Iface   net.Interface
	Kind    string       // "wifi", "ethernet", ...
	Prefix  netip.Prefix // own IPv4 address with mask
	V6      []netip.Prefix
	GW      netip.Addr
	GWMAC   net.HardwareAddr
	NetID   string
	Scoped  bool // NetID is observer-local (gateway unknown)
	SelfMAC string
}

func (n *NetCtx) Vantage() *obs.Vantage {
	return &obs.Vantage{Iface: n.Iface.Name, MAC: n.SelfMAC, Net: n.NetID, Addr: n.Prefix.Addr().String()}
}

func (n *NetCtx) MACID(mac net.HardwareAddr) string { return obs.MACID(mac, n.NetID) }
func (n *NetCtx) IPID(a netip.Addr) string          { return obs.IPID(n.NetID, a) }

// AddrID is IPID for on-link addresses and RoutedIPID otherwise.
func (n *NetCtx) AddrID(a netip.Addr) string {
	if n.Contains(a) {
		return n.IPID(a)
	}
	return obs.RoutedIPID(n.NetID, a)
}

// Contains reports whether addr is on this network (v4 prefix or v6 prefixes).
func (n *NetCtx) Contains(a netip.Addr) bool {
	a = a.Unmap()
	if n.Prefix.Contains(a) {
		return true
	}
	if a.Is6() && a.IsLinkLocalUnicast() {
		return true
	}
	for _, p := range n.V6 {
		if p.Contains(a) {
			return true
		}
	}
	return false
}

// SelfIDs returns identifiers of this host on this network.
func (n *NetCtx) SelfIDs(pub ed25519.PublicKey) []string {
	ids := []string{obs.RKMID(pub), n.MACID(n.Iface.HardwareAddr), n.IPID(n.Prefix.Addr())}
	for _, p := range n.V6 {
		ids = append(ids, n.IPID(p.Addr()))
	}
	return obs.Idents(ids...)
}

func ifaceKind(name string) string {
	if _, err := os.Stat("/sys/class/net/" + name + "/wireless"); err == nil {
		return "wifi"
	}
	if _, err := os.Stat("/sys/class/net/" + name + "/bridge"); err == nil {
		return "bridge"
	}
	return "ethernet"
}

func addrFromIP(ip net.IP) netip.Addr {
	a, _ := netip.AddrFromSlice(ip)
	return a.Unmap()
}

// gatewayMAC reads the kernel neighbor table for the gateway's MAC. If
// provoke is set and the entry is missing, it sends one UDP datagram to make
// the kernel resolve it (an active act; passive mode waits instead).
func gatewayMAC(link netlink.Link, gw netip.Addr, provoke bool) net.HardwareAddr {
	lookup := func() net.HardwareAddr {
		neighs, _ := netlink.NeighList(link.Attrs().Index, unix.AF_INET)
		for _, nb := range neighs {
			if addrFromIP(nb.IP) == gw && len(nb.HardwareAddr) == 6 && nb.State&(unix.NUD_FAILED|unix.NUD_INCOMPLETE) == 0 {
				return nb.HardwareAddr
			}
		}
		return nil
	}
	if m := lookup(); m != nil || !provoke {
		return m
	}
	if c, err := net.Dial("udp4", net.JoinHostPort(gw.String(), "9")); err == nil {
		c.Write([]byte{0})
		c.Close()
	}
	for range 10 {
		time.Sleep(100 * time.Millisecond)
		if m := lookup(); m != nil {
			return m
		}
	}
	return nil
}

// Discover finds attached Ethernet-like networks with IPv4 addresses.
func Discover(pub ed25519.PublicKey, provoke bool) ([]NetCtx, error) {
	links, err := netlink.LinkList()
	if err != nil {
		return nil, err
	}
	routes, _ := netlink.RouteList(nil, unix.AF_INET)
	var out []NetCtx
	for _, l := range links {
		a := l.Attrs()
		if a.Flags&net.FlagUp == 0 || a.Flags&net.FlagLoopback != 0 || len(a.HardwareAddr) != 6 || a.OperState == netlink.OperDown {
			continue
		}
		ifi, err := net.InterfaceByIndex(a.Index)
		if err != nil {
			continue
		}
		addrs, _ := netlink.AddrList(l, unix.AF_UNSPEC)
		nc := NetCtx{Iface: *ifi, Kind: ifaceKind(a.Name), SelfMAC: strings.ToLower(a.HardwareAddr.String())}
		for _, ad := range addrs {
			ip := addrFromIP(ad.IP)
			ones, _ := ad.Mask.Size()
			p := netip.PrefixFrom(ip, ones)
			switch {
			case ip.Is4() && !nc.Prefix.IsValid():
				nc.Prefix = p
			case ip.Is6():
				nc.V6 = append(nc.V6, p)
			}
		}
		if !nc.Prefix.IsValid() {
			continue
		}
		for _, r := range routes {
			if r.LinkIndex == a.Index && r.Gw != nil && (r.Dst == nil || r.Dst.IP.IsUnspecified()) {
				nc.GW = addrFromIP(r.Gw)
				break
			}
		}
		if nc.GW.IsValid() {
			nc.GWMAC = gatewayMAC(l, nc.GW, provoke)
		}
		if nc.GWMAC != nil {
			nc.NetID = obs.NetID(nc.GWMAC, nc.GW, nc.Prefix)
		} else {
			nc.NetID = obs.LocalNetID(pub, a.Name, nc.Prefix)
			nc.Scoped = true
		}
		out = append(out, nc)
	}
	return out, nil
}

// chassisClass reads the SMBIOS chassis type the firmware reports (DMTF
// DSP0134 §7.4.1), falling back to the presence of a battery. "" if unknown.
func chassisClass(sys string) string {
	read := func(p string) string {
		b, _ := os.ReadFile(sys + p)
		return strings.TrimSpace(string(b))
	}
	if v := read("/class/dmi/id/sys_vendor") + " " + read("/class/dmi/id/product_name"); strings.Contains(v, "QEMU") ||
		strings.Contains(v, "VMware") || strings.Contains(v, "VirtualBox") || strings.Contains(v, "KVM") ||
		strings.Contains(v, "Virtual Machine") {
		return "server" // a virtual machine
	}
	switch read("/class/dmi/id/chassis_type") {
	case "3", "4", "5", "6", "7", "15", "16", "24", "35", "36":
		// desktop, low profile, pizza box, mini tower, tower, space-saving,
		// lunch box, sealed-case PC, mini PC, stick PC
		return "desktop"
	case "8", "9", "10", "14", "31", "32":
		// portable, laptop, notebook, sub notebook, convertible, detachable
		return "laptop"
	case "30":
		return "tablet"
	case "17", "23", "25", "28", "29":
		// main server chassis, rack mount, multi-system, blade, blade enclosure
		return "server"
	}
	entries, _ := os.ReadDir(sys + "/class/power_supply")
	for _, e := range entries {
		if read("/class/power_supply/"+e.Name()+"/type") == "Battery" && read("/class/power_supply/"+e.Name()+"/scope") != "Device" {
			return "laptop"
		}
	}
	return ""
}

// Local reports this host, its networks and gateways (mode local).
type Local struct{}

func (Local) Name() string { return "self" }

func (Local) Run(_ context.Context, s *Session, nets []NetCtx) error {
	host, _ := os.Hostname()
	selfIDs := []string{obs.RKMID(s.Pub)}
	claims := []obs.Claim{claim("role", "observer"), claim("os", runtime.GOOS), claim("hostname", host)}
	var u unix.Utsname
	if unix.Uname(&u) == nil {
		claims = append(claims, claim("os.kernel", unix.ByteSliceToString(u.Release[:])))
	}
	if c := chassisClass("/sys"); c != "" {
		claims = append(claims, claimC("class", c, 900))
	}
	for _, n := range nets {
		selfIDs = append(selfIDs, n.MACID(n.Iface.HardwareAddr))
		s.Emit(obs.Body{
			Mode: obs.ModeLocal, Method: "self", Vantage: n.Vantage(),
			Subject: n.SelfIDs(s.Pub),
			Claims:  []obs.Claim{claim("iface", n.Iface.Name+" "+n.Kind+" "+n.SelfMAC)},
		})
		netClaims := []obs.Claim{claim("prefix", n.Prefix.Masked().String()), claim("medium", n.Kind)}
		if n.GW.IsValid() {
			netClaims = append(netClaims, claim("gateway", n.GW.String()))
		}
		if n.Scoped {
			netClaims = append(netClaims, claim("scope", "observer-local"))
		}
		for _, p := range n.V6 {
			if !p.Addr().IsLinkLocalUnicast() {
				netClaims = append(netClaims, claim("prefix6", p.Masked().String()))
			}
		}
		s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "route", Vantage: n.Vantage(),
			Subject: []string{obs.NetEntityID(n.NetID)}, Claims: netClaims})
		if n.GW.IsValid() {
			s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "route", Vantage: n.Vantage(),
				Subject: obs.Idents(n.MACID(n.GWMAC), n.IPID(n.GW)),
				Claims:  []obs.Claim{claim("role", "gateway")}})
		}
	}
	s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "self", Subject: selfIDs, Claims: claims})
	return nil
}
