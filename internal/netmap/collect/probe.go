//go:build linux

package collect

import (
	"bufio"
	"context"
	"errors"
	"net"
	"net/netip"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// DefaultPorts is a small, polite set that identifies common device classes.
var DefaultPorts = []int{22, 23, 53, 80, 139, 443, 445, 515, 554, 631, 1883, 3389, 5000, 5900, 8008, 8080, 8443, 9100, 62078}

// Probe connects to TCP ports on hosts discovered so far, and grabs the SSH
// host key (a strong identifier) and HTTP Server header.
type Probe struct {
	Ports       []int
	Targets     func() map[*NetCtx][]netip.Addr
	Concurrency int
	Timeout     time.Duration
}

func (Probe) Name() string { return "tcp" }

func (c Probe) Run(ctx context.Context, s *Session, _ []NetCtx) error {
	type job struct {
		n    *NetCtx
		ip   netip.Addr
		port int
	}
	jobs := make(chan job)
	var wg sync.WaitGroup
	for range c.Concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				c.probe(ctx, s, j.n, j.ip, j.port)
			}
		}()
	}
	total := 0
	for n, ips := range c.Targets() {
		total += len(ips)
		for _, ip := range ips {
			for _, p := range c.Ports {
				select {
				case jobs <- job{n, ip, p}:
				case <-ctx.Done():
				}
			}
		}
	}
	close(jobs)
	wg.Wait()
	s.Log.Printf("tcp: probed %d hosts × %d ports", total, len(c.Ports))
	return nil
}

func (c Probe) probe(ctx context.Context, s *Session, n *NetCtx, ip netip.Addr, port int) {
	if ctx.Err() != nil {
		return
	}
	addr := netip.AddrPortFrom(ip, uint16(port)).String()
	conn, err := (&net.Dialer{Timeout: c.Timeout}).DialContext(ctx, "tcp", addr)
	if err != nil {
		return
	}
	defer conn.Close()
	claims := []obs.Claim{claim("port", "tcp/"+strconv.Itoa(port))}
	subject := []string{n.IPID(ip)}
	conn.SetDeadline(time.Now().Add(3 * c.Timeout))
	switch port {
	case 22:
		if fp, banner := sshHostKey(conn, addr); fp != "" {
			subject = append(subject, "ssh:"+fp)
			claims = append(claims, claim("ssh.banner", banner))
		}
	case 80, 8080, 8008:
		if srv := httpServer(conn, ip.String()); srv != "" {
			claims = append(claims, claim("http.server", srv))
		}
	}
	s.Emit(obs.Body{Mode: obs.ModeActive, Method: "tcp", Vantage: n.Vantage(), Subject: subject, Claims: claims})
}

var errGotKey = errors.New("host key captured")

type bannerConn struct {
	net.Conn
	buf strings.Builder
}

func (b *bannerConn) Read(p []byte) (int, error) {
	n, err := b.Conn.Read(p)
	if b.buf.Len() < 256 {
		b.buf.Write(p[:n])
	}
	return n, err
}

// sshHostKey runs the SSH handshake only up to key exchange. Host key
// algorithms are listed in a fixed order so that different observers get the
// same key (and therefore the same identifier) from the same server.
func sshHostKey(conn net.Conn, addr string) (fingerprint, banner string) {
	bc := &bannerConn{Conn: conn}
	cfg := &ssh.ClientConfig{
		User: "rakkan",
		HostKeyAlgorithms: []string{ssh.KeyAlgoED25519, ssh.KeyAlgoECDSA256, ssh.KeyAlgoECDSA384,
			ssh.KeyAlgoECDSA521, ssh.KeyAlgoRSASHA512, ssh.KeyAlgoRSASHA256, ssh.KeyAlgoRSA},
		HostKeyCallback: func(_ string, _ net.Addr, key ssh.PublicKey) error {
			fingerprint = ssh.FingerprintSHA256(key)
			return errGotKey
		},
		Timeout: 3 * time.Second,
	}
	if _, _, _, err := ssh.NewClientConn(bc, addr, cfg); err != nil && fingerprint == "" {
		debugSSH(err)
	}
	banner, _, _ = strings.Cut(bc.buf.String(), "\n")
	return fingerprint, strings.TrimSpace(banner)
}

func httpServer(conn net.Conn, host string) string {
	conn.Write([]byte("HEAD / HTTP/1.0\r\nHost: " + host + "\r\nUser-Agent: rakkan\r\n\r\n"))
	r := bufio.NewReader(conn)
	for range 40 {
		l, err := r.ReadString('\n')
		if err != nil {
			return ""
		}
		if k, v, ok := strings.Cut(l, ":"); ok && strings.EqualFold(strings.TrimSpace(k), "server") {
			return strings.TrimSpace(v)
		}
		if strings.TrimSpace(l) == "" {
			return ""
		}
	}
	return ""
}

// RDNS looks up PTR names. rDNS names usually describe the address, not the
// device (dhcp-10-0-0-5.example.net), so they are claims, never identifiers.
type RDNS struct {
	Targets func() map[*NetCtx][]netip.Addr
}

func (RDNS) Name() string { return "rdns" }

func (c RDNS) Run(ctx context.Context, s *Session, _ []NetCtx) error {
	sem := make(chan struct{}, 16)
	var wg sync.WaitGroup
	for n, ips := range c.Targets() {
		for _, ip := range ips {
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer func() { <-sem; wg.Done() }()
				cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
				defer cancel()
				names, err := net.DefaultResolver.LookupAddr(cctx, ip.String())
				if err != nil || len(names) == 0 {
					return
				}
				s.Emit(obs.Body{Mode: obs.ModeActive, Method: "rdns", Vantage: n.Vantage(),
					Subject: []string{n.IPID(ip)}, Claims: []obs.Claim{claimC("rdns", strings.TrimSuffix(names[0], "."), 700)}})
			}()
		}
	}
	wg.Wait()
	return nil
}

var debugSSH = func(error) {}
