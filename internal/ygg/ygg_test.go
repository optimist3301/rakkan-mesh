package ygg

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/net/swarm"
	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
	ma "github.com/multiformats/go-multiaddr"
)

func TestMultiaddr(t *testing.T) {
	key, _, _ := crypto.GenerateEd25519Key(nil)
	sk, err := DeriveKey(key)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := DeriveKey(key)
	if !sk.Equal(again) {
		t.Fatal("the derived key changes between calls")
	}
	pub := sk.Public().(ed25519.PublicKey)
	raw, _ := key.GetPublic().Raw()
	if pub.Equal(ed25519.PublicKey(raw)) {
		t.Fatal("the Yggdrasil key is the libp2p key")
	}
	a, err := ma.NewMultiaddr(Multiaddr(pub).String())
	if err != nil {
		t.Fatal(err)
	}
	got, ok := KeyOf(a)
	if !ok || !got.Equal(pub) {
		t.Fatalf("KeyOf(%s) = %x, %v", a, got, ok)
	}
	if _, ok := KeyOf(a.Encapsulate(ma.StringCast("/p2p/12D3KooWR23RcUWbUjTi2FjpTr8RUEX1PGutXMin3Q4J8qxJPiBf"))); ok {
		t.Fatal("KeyOf accepts a longer address")
	}
	if _, err := ma.NewMultiaddr("/ygg/abcd"); err == nil {
		t.Fatal("a short key parses")
	}
}

// startPair starts two embedded nodes peered over loopback TCP: a listens,
// b connects to it. Nothing else is configured, so they form an overlay of
// their own.
func startPair(t *testing.T) (a, b *Node, ka, kb crypto.PrivKey) {
	t.Helper()
	ka, _, _ = crypto.GenerateEd25519Key(nil)
	kb, _, _ = crypto.GenerateEd25519Key(nil)
	var err error
	if a, err = Start(ka, Config{Peers: []string{}}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { a.Close() })
	addr, err := a.Listen("tcp://127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	if b, err = Start(kb, Config{Peers: []string{"tcp://" + addr.String()}}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { b.Close() })
	deadline := time.Now().Add(10 * time.Second)
	for a.PeersUp() == 0 || b.PeersUp() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the nodes did not peer")
		}
		time.Sleep(50 * time.Millisecond)
	}
	return a, b, ka, kb
}

func newHost(t *testing.T, n *Node, key crypto.PrivKey) host.Host {
	t.Helper()
	h, err := libp2p.New(
		libp2p.Identity(key),
		libp2p.Transport(NewTransport(n)),
		libp2p.ListenAddrs(n.Multiaddr()),
		libp2p.DisableRelay(),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

func TestHostsConnectOverYggdrasil(t *testing.T) {
	a, b, ka, kb := startPair(t)
	ha, hb := newHost(t, a, ka), newHost(t, b, kb)
	if got := ha.Addrs(); len(got) != 1 || !got[0].Equal(a.Multiaddr()) {
		t.Fatalf("addrs %v, want only %s", got, a.Multiaddr())
	}
	if !strings.HasPrefix(a.Multiaddr().String(), "/ygg/") {
		t.Fatalf("multiaddr %s", a.Multiaddr())
	}
	if _, err := ma.NewMultiaddr(a.Multiaddr().String()); err != nil {
		t.Fatal(err)
	}

	const proto = "/test/echo"
	ha.SetStreamHandler(proto, func(s network.Stream) {
		defer s.Close()
		io.Copy(s, s)
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := hb.Connect(ctx, peer.AddrInfo{ID: ha.ID(), Addrs: ha.Addrs()}); err != nil {
		t.Fatal(err)
	}
	s, err := hb.NewStream(ctx, ha.ID(), proto)
	if err != nil {
		t.Fatal(err)
	}
	msg := strings.Repeat("rakkan over yggdrasil ", 20000) // > many packets
	go func() {
		s.Write([]byte(msg))
		s.CloseWrite()
	}()
	got, err := io.ReadAll(s)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != msg {
		t.Fatalf("echo returned %d bytes, want %d", len(got), len(msg))
	}
	conns := hb.Network().ConnsToPeer(ha.ID())
	if len(conns) != 1 || !conns[0].RemoteMultiaddr().Equal(a.Multiaddr()) {
		t.Fatalf("connections %v", conns)
	}
}

// TestOnlineOverlay runs over the real Yggdrasil network: two nodes, each
// peered with different public nodes, find each other by key alone.
// RAKKAN_YGG_ONLINE=1 enables it.
func TestOnlineOverlay(t *testing.T) {
	if os.Getenv("RAKKAN_YGG_ONLINE") == "" {
		t.Skip("set RAKKAN_YGG_ONLINE=1 to use the public Yggdrasil network")
	}
	half := len(DefaultPeers) / 2
	start := func(peers []string) (*Node, crypto.PrivKey) {
		key, _, _ := crypto.GenerateEd25519Key(nil)
		n, err := Start(key, Config{Peers: peers})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { n.Close() })
		return n, key
	}
	a, ka := start(DefaultPeers[:half])
	b, kb := start(DefaultPeers[half:])
	began := time.Now()
	for a.PeersUp() == 0 || b.PeersUp() == 0 {
		if time.Since(began) > 30*time.Second {
			t.Fatalf("peerings up: a %d, b %d", a.PeersUp(), b.PeersUp())
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Logf("peered in %s: a %d, b %d", time.Since(began).Round(time.Millisecond), a.PeersUp(), b.PeersUp())
	ha, hb := newHost(t, a, ka), newHost(t, b, kb)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	began = time.Now()
	if err := b.waitPath(ctx, a.PublicKey()); err != nil {
		t.Fatal(err)
	}
	t.Logf("path found in %s", time.Since(began).Round(time.Millisecond))
	if err := b.waitSession(ctx, a.PublicKey()); err != nil {
		t.Fatal(err)
	}
	t.Logf("session up in %s", time.Since(began).Round(time.Millisecond))
	// Right after peering, the overlay may not route to a yet.
	for {
		err := hb.Connect(ctx, peer.AddrInfo{ID: ha.ID(), Addrs: ha.Addrs()})
		if err == nil {
			break
		}
		if ctx.Err() != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %v", time.Since(began).Round(time.Second), err)
		hb.Network().(interface{ Backoff() *swarm.DialBackoff }).Backoff().Clear(ha.ID())
	}
	t.Logf("connected over the overlay in %s", time.Since(began).Round(time.Millisecond))
	rtt := <-ping.Ping(ctx, hb, ha.ID())
	if rtt.Error != nil {
		t.Fatal(rtt.Error)
	}
	t.Logf("ping %s", rtt.RTT)

	ha.SetStreamHandler("/test/sink", func(s network.Stream) {
		defer s.Close()
		n, _ := io.Copy(io.Discard, s)
		fmt.Fprintf(s, "%d", n)
	})
	s, err := hb.NewStream(ctx, ha.ID(), "/test/sink")
	if err != nil {
		t.Fatal(err)
	}
	const size = 4 << 20
	began = time.Now()
	if _, err := s.Write(make([]byte, size)); err != nil {
		t.Fatal(err)
	}
	s.CloseWrite()
	ack, err := io.ReadAll(s)
	if err != nil || string(ack) != fmt.Sprint(size) {
		t.Fatalf("sink got %q, %v", ack, err)
	}
	el := time.Since(began)
	t.Logf("sent %d MiB in %s: %.2f Mbit/s", size>>20, el.Round(time.Millisecond), float64(size)*8/el.Seconds()/1e6)
}
