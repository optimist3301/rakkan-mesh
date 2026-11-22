//go:build linux

package collect

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net/netip"
	"strconv"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// Trace discovers the L3 path to explicitly given destinations, without
// privileges: a UDP socket with IP_RECVERR receives the ICMP time-exceeded
// errors via the socket error queue (the technique of tracepath(8)).
// Each hop becomes an observation linked (link.l3) to the previous one, so
// paths from different observers merge where they share routers.
type Trace struct {
	Targets []netip.Addr
	MaxHops int
	Timeout time.Duration // per hop
}

func (Trace) Name() string { return "trace" }

const tracePortBase = 33434

type hop struct {
	ttl     int
	addr    netip.Addr // invalid = no answer
	rtt     time.Duration
	reached bool // the destination answered (port unreachable)
	end     bool // a router declared the destination unreachable
}

func (c Trace) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	var errs []error
	for _, dst := range c.Targets {
		if ctx.Err() != nil {
			break
		}
		if !dst.Is4() {
			errs = append(errs, fmt.Errorf("trace %s: only IPv4 for now", dst))
			continue
		}
		hops, err := traceUDP(ctx, dst, c.MaxHops, c.Timeout)
		if err != nil {
			errs = append(errs, fmt.Errorf("trace %s: %w", dst, err))
			continue
		}
		n := routeNet(nets, hops)
		emitPath(s, n, dst, hops)
		answered := 0
		for _, h := range hops {
			if h.addr.IsValid() {
				answered++
			}
		}
		s.Log.Printf("trace %s: %d hops, %d answered, reached=%v", dst, len(hops), answered, len(hops) > 0 && hops[len(hops)-1].reached)
	}
	return errors.Join(errs...)
}

// routeNet picks the attached network the path leaves through: the one whose
// gateway answered as the first hop (or the first network).
func routeNet(nets []NetCtx, hops []hop) *NetCtx {
	if len(hops) > 0 && hops[0].addr.IsValid() {
		for i := range nets {
			if nets[i].Contains(hops[0].addr) {
				return &nets[i]
			}
		}
	}
	return &nets[0]
}

// emitPath records hops and the links between consecutive answering hops.
// Silent hops are bridged with a gap count, never invented.
func emitPath(s *Session, n *NetCtx, dst netip.Addr, hops []hop) {
	prev := obs.RKMID(s.Pub) // the path starts at this host
	gap := 0
	for _, h := range hops {
		if !h.addr.IsValid() {
			gap++
			continue
		}
		id := n.AddrID(h.addr)
		claims := []obs.Claim{
			claimC("link.l3", prev, linkConf(gap)),
			claim("trace.ttl", strconv.Itoa(h.ttl)),
		}
		if gap > 0 {
			claims = append(claims, claim("trace.gap", strconv.Itoa(gap)))
		}
		if h.reached {
			claims = append(claims, claim("trace.target", dst.String()))
		} else {
			claims = append(claims, claim("role", "router"))
		}
		s.Emit(obs.Body{Mode: obs.ModeActive, Method: "trace", Vantage: n.Vantage(), Subject: []string{id}, Claims: claims})
		prev, gap = id, 0
	}
}

// linkConf lowers confidence of links that skip silent hops.
func linkConf(gap int) uint16 {
	if gap == 0 {
		return 1000
	}
	return uint16(max(200, 900-200*gap))
}

func traceUDP(ctx context.Context, dst netip.Addr, maxHops int, timeout time.Duration) ([]hop, error) {
	fd, err := unix.Socket(unix.AF_INET, unix.SOCK_DGRAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	defer unix.Close(fd)
	if err := unix.SetsockoptInt(fd, unix.IPPROTO_IP, unix.IP_RECVERR, 1); err != nil {
		return nil, err
	}
	var hops []hop
	silent := 0
	for ttl := 1; ttl <= maxHops && ctx.Err() == nil; ttl++ {
		if err := unix.SetsockoptInt(fd, unix.IPPROTO_IP, unix.IP_TTL, ttl); err != nil {
			return hops, err
		}
		port := tracePortBase + ttl
		start := time.Now()
		if err := unix.Sendto(fd, []byte("rakkan-trace"), 0, &unix.SockaddrInet4{Port: port, Addr: dst.As4()}); err != nil &&
			!errors.Is(err, unix.EHOSTUNREACH) && !errors.Is(err, unix.ECONNREFUSED) {
			return hops, err
		}
		h := hop{ttl: ttl}
		deadline := start.Add(timeout)
		for time.Now().Before(deadline) {
			ms := int(time.Until(deadline).Milliseconds()) + 1
			fds := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLERR | unix.POLLIN}}
			if n, err := unix.Poll(fds, ms); err != nil && !errors.Is(err, unix.EINTR) || n == 0 {
				break
			}
			from, gotPort, kind, ok := readErrQueue(fd)
			if !ok {
				// Not an error-queue message (e.g. a stray datagram): drain it.
				buf := make([]byte, 512)
				unix.Recvfrom(fd, buf, unix.MSG_DONTWAIT)
				continue
			}
			if gotPort != port { // late answer to an earlier probe
				continue
			}
			h.addr, h.rtt = from, time.Since(start)
			h.reached, h.end = kind == errDestination, kind == errUnreachable
			break
		}
		hops = append(hops, h)
		if h.reached || h.end {
			break
		}
		if h.addr.IsValid() {
			silent = 0
		} else if silent++; silent >= 5 {
			break // the rest of the path is filtered
		}
	}
	// Trim trailing silence: it says nothing about topology.
	for len(hops) > 0 && !hops[len(hops)-1].addr.IsValid() {
		hops = hops[:len(hops)-1]
	}
	return hops, nil
}

type errKind int

const (
	errTimeExceeded errKind = iota // an intermediate router
	errDestination                 // port unreachable: the destination itself
	errUnreachable                 // net/host/admin unreachable: the path ends here
)

// readErrQueue reads one ICMP error: the offending router's address, the
// destination port of the probe it answers, and what kind of answer it is.
func readErrQueue(fd int) (from netip.Addr, port int, kind errKind, ok bool) {
	buf := make([]byte, 512)
	oob := make([]byte, 512)
	_, oobn, _, sa, err := unix.Recvmsg(fd, buf, oob, unix.MSG_ERRQUEUE|unix.MSG_DONTWAIT)
	if err != nil {
		return
	}
	if dst, isV4 := sa.(*unix.SockaddrInet4); isV4 {
		port = dst.Port
	}
	cmsgs, err := unix.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		return
	}
	for _, c := range cmsgs {
		if c.Header.Level != unix.IPPROTO_IP || c.Header.Type != unix.IP_RECVERR {
			continue
		}
		return parseRecvErr(c.Data, port)
	}
	return
}

// parseRecvErr decodes struct sock_extended_err followed by the offender's
// struct sockaddr_in (SO_EE_OFFENDER).
func parseRecvErr(data []byte, port int) (from netip.Addr, _ int, kind errKind, ok bool) {
	const eeSize = int(unsafe.Sizeof(unix.SockExtendedErr{}))
	if len(data) < eeSize+8 {
		return
	}
	ee := unix.SockExtendedErr{
		Errno: binary.NativeEndian.Uint32(data[0:4]), Origin: data[4], Type: data[5], Code: data[6],
	}
	if ee.Origin != unix.SO_EE_ORIGIN_ICMP {
		return
	}
	off := data[eeSize:]
	if binary.NativeEndian.Uint16(off[0:2]) != unix.AF_INET {
		return
	}
	from = netip.AddrFrom4([4]byte(off[4:8]))
	switch {
	case ee.Type == 11:
		kind = errTimeExceeded
	case ee.Type == 3 && ee.Code == 3:
		kind = errDestination
	case ee.Type == 3:
		kind = errUnreachable
	default:
		return netip.Addr{}, port, 0, false
	}
	return from, port, kind, true
}
