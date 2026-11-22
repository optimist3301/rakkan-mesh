//go:build linux

package collect

import (
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"net/netip"
	"strconv"
	"sync"
	"time"

	"github.com/vishvananda/netlink"
	"golang.org/x/net/icmp"
	"golang.org/x/net/ipv4"
	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// Neigh reads the kernel ARP/NDP cache. The mode says who caused the entries:
// ModeActive right after a sweep, ModePassive while watching.
type Neigh struct{ Mode obs.Mode }

func (Neigh) Name() string { return "neigh" }

func (c Neigh) Run(_ context.Context, s *Session, nets []NetCtx) error {
	for i := range nets {
		n := &nets[i]
		list, err := netlink.NeighList(n.Iface.Index, unix.AF_UNSPEC)
		if err != nil {
			return err
		}
		for _, nb := range list {
			if len(nb.HardwareAddr) != 6 || nb.State&(unix.NUD_REACHABLE|unix.NUD_STALE|unix.NUD_DELAY|unix.NUD_PROBE|unix.NUD_PERMANENT) == 0 {
				continue
			}
			ip := addrFromIP(nb.IP)
			if !n.Contains(ip) {
				continue
			}
			conf := uint16(1000)
			if nb.State&unix.NUD_STALE != 0 {
				conf = 800 // binding may have changed since last confirmation
			}
			var claims []obs.Claim
			if nb.Flags&netlink.NTF_ROUTER != 0 {
				claims = append(claims, claim("role", "router"))
			}
			s.Emit(obs.Body{
				Mode: c.Mode, Method: "neigh", Vantage: n.Vantage(),
				Subject: obs.Idents(n.MACID(nb.HardwareAddr), n.IPID(ip)),
				Claims:  append(claims, claimC("seen.l2", "neigh", conf)),
			})
		}
	}
	return nil
}

// SweepTargets returns the IPv4 hosts to probe on a network. Networks larger
// than maxHosts are reduced to the /24 around our own address.
func SweepTargets(n *NetCtx, override netip.Prefix, maxHosts int) []netip.Addr {
	p := n.Prefix.Masked()
	if override.IsValid() {
		p = override.Masked()
	} else if 1<<(32-p.Bits()) > maxHosts {
		p = netip.PrefixFrom(n.Prefix.Addr(), 24).Masked()
	}
	var out []netip.Addr
	start := binary.BigEndian.Uint32(p.Addr().AsSlice())
	size := uint32(1) << (32 - p.Bits())
	for i := uint32(1); i+1 < size; i++ { // skip network and broadcast addresses
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], start+i)
		a := netip.AddrFrom4(b)
		if a != n.Prefix.Addr() {
			out = append(out, a)
		}
	}
	return out
}

// Sweep pings every target (unprivileged ICMP sockets, falling back to UDP
// datagrams) so that the kernel resolves their MACs, then reads the cache.
// Hosts that drop ICMP still answer ARP, so they show up via Neigh.
type Sweep struct {
	CIDR     netip.Prefix
	MaxHosts int
	Rate     time.Duration // delay between probes
}

func (Sweep) Name() string { return "ping" }

func (c Sweep) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	for i := range nets {
		n := &nets[i]
		if c.CIDR.IsValid() && !n.Prefix.Masked().Overlaps(c.CIDR) {
			continue
		}
		targets := SweepTargets(n, c.CIDR, c.MaxHosts)
		s.Log.Printf("ping: sweeping %d hosts on %s", len(targets), n.Iface.Name)
		if err := c.ping(ctx, s, n, targets); err != nil {
			s.Log.Printf("ping: ICMP unavailable (%v), using UDP probes", err)
			udpTouch(ctx, targets, c.Rate)
		}
	}
	time.Sleep(1500 * time.Millisecond) // let ARP resolution finish
	return Neigh{Mode: obs.ModeActive}.Run(ctx, s, nets)
}

func udpTouch(ctx context.Context, targets []netip.Addr, rate time.Duration) {
	for _, t := range targets {
		if ctx.Err() != nil {
			return
		}
		if c, err := net.Dial("udp4", netip.AddrPortFrom(t, 9).String()); err == nil {
			c.Write([]byte{0})
			c.Close()
		}
		time.Sleep(rate)
	}
}

func (c Sweep) ping(ctx context.Context, s *Session, n *NetCtx, targets []netip.Addr) error {
	conn, err := icmp.ListenPacket("udp4", n.Prefix.Addr().String())
	if err != nil {
		return err
	}
	defer conn.Close()
	pc := conn.IPv4PacketConn()
	_ = pc.SetControlMessage(ipv4.FlagTTL, true)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		buf := make([]byte, 1500)
		seen := map[netip.Addr]bool{}
		for {
			nr, cm, src, err := pc.ReadFrom(buf)
			if err != nil {
				return
			}
			m, err := icmp.ParseMessage(1, buf[:nr])
			if err != nil || m.Type != ipv4.ICMPTypeEchoReply {
				continue
			}
			ua, ok := src.(*net.UDPAddr)
			if !ok {
				continue
			}
			ip := addrFromIP(ua.IP)
			if seen[ip] || !n.Contains(ip) {
				continue
			}
			seen[ip] = true
			claims := []obs.Claim{claim("alive", "icmp")}
			if cm != nil && cm.TTL > 0 {
				claims = append(claims, claim("ttl", strconv.Itoa(cm.TTL)), claimC("os.family", ttlFamily(cm.TTL), 400))
			}
			s.Emit(obs.Body{Mode: obs.ModeActive, Method: "ping", Vantage: n.Vantage(),
				Subject: []string{n.IPID(ip)}, Claims: claims})
		}
	}()

	for i, t := range targets {
		if ctx.Err() != nil {
			break
		}
		msg := icmp.Message{Type: ipv4.ICMPTypeEcho, Body: &icmp.Echo{Seq: i & 0xffff, Data: []byte("rakkan")}}
		b, _ := msg.Marshal(nil)
		if _, err := conn.WriteTo(b, &net.UDPAddr{IP: t.AsSlice()}); err != nil && i == 0 {
			return fmt.Errorf("send: %w", err)
		}
		time.Sleep(c.Rate)
	}
	time.Sleep(1500 * time.Millisecond)
	conn.SetReadDeadline(time.Now())
	wg.Wait()
	return nil
}

// ttlFamily guesses the OS family from an initial TTL (low confidence).
func ttlFamily(ttl int) string {
	switch {
	case ttl <= 64:
		return "unix"
	case ttl <= 128:
		return "windows"
	default:
		return "network-device"
	}
}
