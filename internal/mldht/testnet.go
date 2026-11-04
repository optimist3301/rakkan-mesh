package mldht

import (
	"bytes"
	"crypto/rand"
	"net"
	"slices"
	"sync"
)

// TestNetwork is a DHT for tests, on loopback: nodes that know each
// other, answer find_node and get with the closest of them and what they
// store, and take verified puts, as Mainline's nodes do.
type TestNetwork struct {
	nodes []*fakeNode
}

// NewTestNetwork starts n nodes.
func NewTestNetwork(n int) (*TestNetwork, error) {
	tn := &TestNetwork{}
	for range n {
		conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
		if err != nil {
			tn.Close()
			return nil, err
		}
		f := &fakeNode{conn: conn, all: &tn.nodes, store: map[[20]byte]map[string]any{}}
		rand.Read(f.id[:])
		tn.nodes = append(tn.nodes, f)
	}
	for _, f := range tn.nodes {
		go f.serve()
	}
	return tn, nil
}

// Bootstrap is a few of the nodes' addresses, to start lookups from.
func (tn *TestNetwork) Bootstrap() []string {
	var out []string
	for _, f := range tn.nodes[:min(3, len(tn.nodes))] {
		out = append(out, f.addr())
	}
	return out
}

// Close stops the nodes.
func (tn *TestNetwork) Close() {
	for _, f := range tn.nodes {
		f.conn.Close()
	}
}

type fakeNode struct {
	id    [20]byte
	conn  net.PacketConn
	all   *[]*fakeNode
	mu    sync.Mutex
	store map[[20]byte]map[string]any
	puts  int
}

func (f *fakeNode) addr() string { return f.conn.LocalAddr().String() }

func (f *fakeNode) token(from net.Addr) []byte {
	h := sha1Of(f.id[:], []byte(from.String()))
	return h[:8]
}

func (f *fakeNode) serve() {
	buf := make([]byte, 2048)
	for {
		n, from, err := f.conn.ReadFrom(buf)
		if err != nil {
			return
		}
		v, err := decodeAll(bytes.Clone(buf[:n]))
		msg := dictOf(v)
		if err != nil || string(bytesAt(msg, "y")) != "q" {
			continue
		}
		a := dictOf(msg["a"])
		reply := func(r map[string]any) {
			r["id"] = f.id[:]
			f.conn.WriteTo(encode(nil, map[string]any{"t": bytesAt(msg, "t"), "y": "r", "r": r}), from)
		}
		fail := func(code int, m string) {
			f.conn.WriteTo(encode(nil, map[string]any{"t": bytesAt(msg, "t"), "y": "e", "e": []any{code, m}}), from)
		}
		switch string(bytesAt(msg, "q")) {
		case "find_node":
			var target [20]byte
			copy(target[:], bytesAt(a, "target"))
			reply(map[string]any{"nodes": f.closest(target)})
		case "get":
			var target [20]byte
			copy(target[:], bytesAt(a, "target"))
			r := map[string]any{"token": f.token(from), "nodes": f.closest(target)}
			f.mu.Lock()
			for k, v := range f.store[target] {
				r[k] = v
			}
			f.mu.Unlock()
			reply(r)
		case "put":
			if !bytes.Equal(bytesAt(a, "token"), f.token(from)) {
				fail(203, "bad token")
				continue
			}
			seq, _ := intAt(a, "seq")
			val, _ := a["v"].([]byte)
			it := &Item{Key: bytesAt(a, "k"), Salt: bytesAt(a, "salt"), Seq: seq, Value: val, Sig: bytesAt(a, "sig")}
			if it.Verify() != nil {
				fail(206, "bad signature")
				continue
			}
			target := Target(it.Key, it.Salt)
			f.mu.Lock()
			if old, ok := intAt(f.store[target], "seq"); ok && old > seq {
				f.mu.Unlock()
				fail(302, "sequence number less than current")
				continue
			}
			f.store[target] = map[string]any{"k": []byte(it.Key), "seq": seq, "sig": it.Sig, "v": val}
			f.puts++
			f.mu.Unlock()
			reply(map[string]any{})
		}
	}
}

// closest are the 8 nodes closest to target, compact.
func (f *fakeNode) closest(target [20]byte) []byte {
	nodes := slices.Clone(*f.all)
	slices.SortFunc(nodes, func(a, b *fakeNode) int {
		if closer(target, a.id, b.id) {
			return -1
		}
		return 1
	})
	var out []byte
	for _, n := range nodes[:min(8, len(nodes))] {
		ua := n.conn.LocalAddr().(*net.UDPAddr)
		out = append(out, n.id[:]...)
		out = append(out, ua.IP.To4()...)
		out = append(out, byte(ua.Port>>8), byte(ua.Port))
	}
	return out
}
