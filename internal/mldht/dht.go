// Package mldht reads and writes signed records in the BitTorrent Mainline
// DHT: a client of KRPC (BEP 5) that finds the nodes closest to a key and
// gets or puts mutable items (BEP 44), Ed25519-signed values of up to
// 1000 bytes stored by whichever nodes are closest to the hash of the
// public key and a salt. The DHT has millions of nodes and no operator:
// nobody can take a record down, and nobody but the key's holder can
// change it. Anyone can store a signed item again, so others keep it
// alive.
//
// The client stores nothing for others and asks not to be (BEP 43's
// read-only flag); IPv4 only.
package mldht

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"errors"
	"fmt"
	"net"
	"slices"
	"sync"
	"time"
)

// DefaultBootstrap are well-known entry points of the Mainline DHT.
var DefaultBootstrap = []string{
	"router.bittorrent.com:6881",
	"dht.transmissionbt.com:6881",
	"router.utorrent.com:6881",
	"dht.libtorrent.org:25401",
}

// Client queries the DHT.
type Client struct {
	conn      net.PacketConn
	id        [20]byte
	bootstrap []string
	timeout   time.Duration // of one query

	mu      sync.Mutex
	pending map[string]*pending
	next    uint32
	done    chan struct{}
}

type pending struct {
	addr string
	ch   chan map[string]any
}

// Listen opens a client on a UDP address (":0" for any port) with the
// bootstrap nodes it starts lookups from (host:port).
func Listen(addr string, bootstrap []string) (*Client, error) {
	conn, err := net.ListenPacket("udp4", addr)
	if err != nil {
		return nil, err
	}
	c := &Client{conn: conn, bootstrap: bootstrap, timeout: time.Second, pending: map[string]*pending{}, done: make(chan struct{})}
	rand.Read(c.id[:])
	go c.readLoop()
	return c, nil
}

// Close stops the client.
func (c *Client) Close() error {
	select {
	case <-c.done:
	default:
		close(c.done)
	}
	return c.conn.Close()
}

// readLoop hands responses to their queries: by transaction ID, and only
// from the address queried. Queries to the client go unanswered.
func (c *Client) readLoop() {
	buf := make([]byte, 2048)
	for {
		n, from, err := c.conn.ReadFrom(buf)
		if err != nil {
			return
		}
		// The decoded strings point into the packet: a copy of its own.
		v, err := decodeAll(bytes.Clone(buf[:n]))
		msg := dictOf(v)
		if err != nil || msg == nil {
			continue
		}
		t := string(bytesAt(msg, "t"))
		c.mu.Lock()
		p := c.pending[t]
		if p != nil && p.addr == from.String() {
			delete(c.pending, t)
		} else {
			p = nil
		}
		c.mu.Unlock()
		if p != nil {
			p.ch <- msg
		}
	}
}

// errRemote is a KRPC error the node answered with.
type errRemote struct {
	code int64
	msg  string
}

func (e errRemote) Error() string { return fmt.Sprintf("mldht: node error %d: %s", e.code, e.msg) }

// query sends a query and waits for its response's "r" dictionary.
func (c *Client) query(ctx context.Context, addr *net.UDPAddr, q string, args map[string]any) (map[string]any, error) {
	args["id"] = c.id[:]
	c.mu.Lock()
	c.next++
	t := string([]byte{byte(c.next >> 8), byte(c.next)})
	p := &pending{addr: addr.String(), ch: make(chan map[string]any, 1)}
	c.pending[t] = p
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		if c.pending[t] == p {
			delete(c.pending, t)
		}
		c.mu.Unlock()
	}()
	msg := encode(nil, map[string]any{"t": t, "y": "q", "q": q, "a": args, "ro": 1})
	if _, err := c.conn.WriteTo(msg, addr); err != nil {
		return nil, err
	}
	timer := time.NewTimer(c.timeout)
	defer timer.Stop()
	select {
	case r := <-p.ch:
		switch string(bytesAt(r, "y")) {
		case "r":
			if d := dictOf(r["r"]); d != nil {
				return d, nil
			}
		case "e":
			if e, ok := r["e"].([]any); ok && len(e) == 2 {
				code, _ := e[0].(int64)
				m, _ := e[1].([]byte)
				return nil, errRemote{code, string(m)}
			}
		}
		return nil, errors.New("mldht: malformed response")
	case <-timer.C:
		return nil, errors.New("mldht: timeout")
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.done:
		return nil, net.ErrClosed
	}
}

// contact is a node: its ID (zero before it answered) and address.
type contact struct {
	id    [20]byte
	known bool
	boot  bool // a bootstrap node
	addr  *net.UDPAddr
}

// answer is what a node closest to a lookup's target said.
type answer struct {
	contact
	r     map[string]any
	token []byte
}

// lookup parameters: nodes asked at once, the closest kept.
const (
	alpha      = 8
	kClosest   = 8
	maxQueries = 300
)

func distance(a, b [20]byte) (d [20]byte) {
	for i := range d {
		d[i] = a[i] ^ b[i]
	}
	return d
}

func closer(target, a, b [20]byte) bool {
	da, db := distance(a, target), distance(b, target)
	return slices.Compare(da[:], db[:]) < 0
}

// lookup asks nodes ever closer to target with q (a "get"), and returns
// the answers of the closest that answered, closest first.
func (c *Client) lookup(ctx context.Context, target [20]byte, q string, args func() map[string]any) ([]answer, error) {
	var cands []*contact
	seen := map[string]bool{}
	add := func(ct *contact) {
		if seen[ct.addr.String()] {
			return
		}
		seen[ct.addr.String()] = true
		cands = append(cands, ct)
	}
	for _, b := range c.bootstrap {
		addrs, err := net.DefaultResolver.LookupHost(ctx, hostOf(b))
		if err != nil {
			continue
		}
		for _, a := range addrs {
			if ua, err := net.ResolveUDPAddr("udp4", net.JoinHostPort(a, portOf(b))); err == nil && ua.IP.To4() != nil {
				add(&contact{addr: ua, boot: true})
			}
		}
	}
	if len(cands) == 0 {
		return nil, errors.New("mldht: no bootstrap node resolves")
	}
	queried, failed := map[*contact]bool{}, map[*contact]bool{}
	var answers []answer
	type result struct {
		ct  *contact
		r   map[string]any
		err error
	}
	results := make(chan result)
	inFlight, queries, retries := 0, 0, 2
	// next is the closest candidate not yet asked, if it can still be among
	// the kClosest: unknown nodes (bootstrap) first.
	next := func() *contact {
		cands = slices.DeleteFunc(cands, func(ct *contact) bool { return queried[ct] && (failed[ct] || !ct.known) })
		slices.SortStableFunc(cands, func(a, b *contact) int {
			switch {
			case !a.known && b.known:
				return -1
			case a.known && !b.known:
				return 1
			case !a.known:
				return 0
			case closer(target, a.id, b.id):
				return -1
			case closer(target, b.id, a.id):
				return 1
			}
			return 0
		})
		known := 0
		for _, ct := range cands {
			if ct.known {
				if known++; known > kClosest {
					return nil
				}
			}
			if !queried[ct] {
				return ct
			}
		}
		return nil
	}
	for {
		for inFlight < alpha && queries < maxQueries {
			ct := next()
			if ct == nil {
				break
			}
			queried[ct] = true
			queries++
			inFlight++
			go func() {
				// Bootstrap routers keep no items and may answer only
				// find_node (BEP 5): they are asked that.
				var r map[string]any
				var err error
				if ct.boot {
					r, err = c.query(ctx, ct.addr, "find_node", map[string]any{"target": target[:]})
				} else {
					r, err = c.query(ctx, ct.addr, q, args())
				}
				results <- result{ct, r, err}
			}()
		}
		if inFlight == 0 && len(answers) == 0 && retries > 0 {
			// Nothing answered: the bootstrap nodes once more.
			retries--
			for ct := range queried {
				if ct.boot {
					delete(queried, ct)
					delete(failed, ct)
					ct.known = false
					cands = append(cands, ct)
				}
			}
			continue
		}
		if inFlight == 0 {
			break
		}
		res := <-results
		inFlight--
		if res.err != nil {
			failed[res.ct] = true
			continue
		}
		if id := bytesAt(res.r, "id"); len(id) == 20 {
			copy(res.ct.id[:], id)
			res.ct.known = true
		} else {
			failed[res.ct] = true
			continue
		}
		if !res.ct.boot {
			answers = append(answers, answer{contact: *res.ct, r: res.r, token: bytesAt(res.r, "token")})
		}
		for _, n := range parseNodes(bytesAt(res.r, "nodes")) {
			add(n)
		}
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	answers = slices.DeleteFunc(answers, func(a answer) bool { return !a.known })
	slices.SortFunc(answers, func(a, b answer) int {
		if closer(target, a.id, b.id) {
			return -1
		}
		if closer(target, b.id, a.id) {
			return 1
		}
		return 0
	})
	return answers, nil
}

// parseNodes reads compact IPv4 node infos: 20 bytes of ID, 4 of address,
// 2 of port each.
func parseNodes(b []byte) []*contact {
	var out []*contact
	for ; len(b) >= 26; b = b[26:] {
		ct := &contact{known: true, addr: &net.UDPAddr{IP: net.IP(slices.Clone(b[20:24])), Port: int(b[24])<<8 | int(b[25])}}
		copy(ct.id[:], b[:20])
		if ct.addr.Port == 0 || ct.addr.IP.IsUnspecified() {
			continue
		}
		out = append(out, ct)
	}
	return out
}

func hostOf(hp string) string {
	h, _, err := net.SplitHostPort(hp)
	if err != nil {
		return hp
	}
	return h
}

func portOf(hp string) string {
	_, p, err := net.SplitHostPort(hp)
	if err != nil {
		return "6881"
	}
	return p
}

// sha1Of is SHA-1 of the concatenation of parts.
func sha1Of(parts ...[]byte) (out [20]byte) {
	h := sha1.New()
	for _, p := range parts {
		h.Write(p)
	}
	copy(out[:], h.Sum(nil))
	return out
}
