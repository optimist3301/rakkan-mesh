package mldht

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"net"
	"os"
	"slices"
	"sync"
	"testing"
	"time"
)

func TestBencode(t *testing.T) {
	v := map[string]any{"a": int64(-3), "b": []byte("xyz"), "c": []any{int64(0), []byte(""), map[string]any{}}}
	b := encode(nil, v)
	if string(b) != "d1:ai-3e1:b3:xyz1:cli0e0:deee" {
		t.Fatalf("%q", b)
	}
	got, err := decodeAll(b)
	if err != nil {
		t.Fatal(err)
	}
	if again := encode(nil, got); !bytes.Equal(again, b) {
		t.Fatalf("round trip %q", again)
	}
	for _, bad := range []string{"", "i", "ie", "i-0e", "i03e", "03:abc", "5:abc", "l", "d1:ai1ee1", "d1:ai1e1:ai2ee", "di1ei2ee", "x",
		string(bytes.Repeat([]byte("l"), maxDepth+2)) + string(bytes.Repeat([]byte("e"), maxDepth+2))} {
		if _, err := decodeAll([]byte(bad)); err == nil {
			t.Errorf("%q decoded", bad)
		}
	}
}

func TestItemSignature(t *testing.T) {
	_, key, _ := ed25519.GenerateKey(rand.Reader)
	it, err := Sign(key, []byte("salt"), 7, []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	if err := it.Verify(); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Item){
		func(i *Item) { i.Seq++ },
		func(i *Item) { i.Value = []byte("hellO") },
		func(i *Item) { i.Salt = []byte("salT") },
	} {
		c := *it
		change(&c)
		if c.Verify() == nil {
			t.Error("a changed item verified")
		}
	}
	if string(signed([]byte("ab"), 1, []byte("v"))) != "4:salt2:ab3:seqi1e1:v1:v" {
		t.Errorf("%q", signed([]byte("ab"), 1, []byte("v")))
	}
}

// fakeNode is a DHT node for tests: it knows all the others, answers get
// with the closest of them and what it stores, and takes verified puts.
type fakeNode struct {
	id    [20]byte
	conn  net.PacketConn
	all   *[]*fakeNode
	mu    sync.Mutex
	store map[[20]byte]map[string]any
	puts  int
}

func fakeNet(t *testing.T, n int) []*fakeNode {
	t.Helper()
	var nodes []*fakeNode
	for range n {
		conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		f := &fakeNode{conn: conn, all: &nodes, store: map[[20]byte]map[string]any{}}
		rand.Read(f.id[:])
		nodes = append(nodes, f)
		t.Cleanup(func() { conn.Close() })
	}
	for _, f := range nodes {
		go f.serve()
	}
	return nodes
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

// An item put through one client is got through another, the newer of two
// wins, and only the nodes closest to its target store it.
func TestPutGet(t *testing.T) {
	nodes := fakeNet(t, 60)
	boot := []string{nodes[0].addr(), nodes[1].addr()}
	pub, key, _ := ed25519.GenerateKey(rand.Reader)
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	a, err := Listen("127.0.0.1:0", boot)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := Listen("127.0.0.1:0", boot[1:])
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if _, err := b.Get(ctx, pub, []byte("s"), 0); err != ErrNotFound {
		t.Fatalf("before any put: %v", err)
	}
	for seq, v := range []string{"first", "second"} {
		it, _ := Sign(key, []byte("s"), int64(seq+1), []byte(v))
		n, err := a.Put(ctx, it)
		if err != nil || n != 8 {
			t.Fatalf("put %s: %d nodes, %v", v, n, err)
		}
	}
	got, err := b.Get(ctx, pub, []byte("s"), 0)
	if err != nil || string(got.Value) != "second" || got.Seq != 2 {
		t.Fatalf("got %+v, %v", got, err)
	}
	// An older item is refused; another salt is another item.
	old, _ := Sign(key, []byte("s"), 1, []byte("first"))
	if _, err := a.Put(ctx, old); err == nil {
		t.Error("an older item was stored")
	}
	if _, err := b.Get(ctx, pub, []byte("t"), 0); err != ErrNotFound {
		t.Errorf("another salt: %v", err)
	}
	// The 8 storing nodes are the closest to the target.
	target := Target(pub, []byte("s"))
	sorted := slices.Clone(nodes)
	slices.SortFunc(sorted, func(x, y *fakeNode) int {
		if closer(target, x.id, y.id) {
			return -1
		}
		return 1
	})
	for i, n := range sorted {
		n.mu.Lock()
		_, has := n.store[target]
		n.mu.Unlock()
		if has != (i < 8) {
			t.Errorf("node %d from the target: stores %v", i, has)
		}
	}
}

// A forged item, or one for another key, is not taken.
func TestGetRefusesForgeries(t *testing.T) {
	nodes := fakeNet(t, 20)
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	_, other, _ := ed25519.GenerateKey(rand.Reader)
	target := Target(pub, nil)
	bad, _ := Sign(other, nil, 5, []byte("forged"))
	for _, n := range nodes {
		n.mu.Lock()
		n.store[target] = map[string]any{"k": []byte(pub), "seq": int64(5), "sig": bad.Sig, "v": []byte("forged")}
		n.mu.Unlock()
	}
	c, err := Listen("127.0.0.1:0", []string{nodes[0].addr()})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if it, err := c.Get(t.Context(), pub, nil, 0); err != ErrNotFound {
		t.Fatalf("got %+v, %v", it, err)
	}
}

// The real Mainline DHT takes an item and gives it back
// (RAKKAN_MLDHT_ONLINE=1): its nodes check BEP 44's signature, so this
// checks the format.
func TestOnline(t *testing.T) {
	if os.Getenv("RAKKAN_MLDHT_ONLINE") == "" {
		t.Skip("RAKKAN_MLDHT_ONLINE not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pub, key, _ := ed25519.GenerateKey(rand.Reader)
	salt := []byte("rakkan-mesh/test")
	value := []byte(fmt.Sprintf("hello from a test at %v", time.Now().UTC()))
	put, err := Listen(":0", DefaultBootstrap)
	if err != nil {
		t.Fatal(err)
	}
	defer put.Close()
	it, _ := Sign(key, salt, time.Now().Unix(), value)
	start := time.Now()
	n, err := put.Put(ctx, it)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("stored on %d nodes in %v", n, time.Since(start))
	get, err := Listen(":0", DefaultBootstrap)
	if err != nil {
		t.Fatal(err)
	}
	defer get.Close()
	start = time.Now()
	got, err := get.Get(ctx, pub, salt, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Value, value) {
		t.Fatalf("got %q", got.Value)
	}
	t.Logf("got back in %v", time.Since(start))
}
