package resolve

import (
	"bytes"
	"crypto/ed25519"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

const net1 = "nnet1"

var t0 = time.Date(2026, 11, 1, 12, 0, 0, 0, time.UTC).UnixMilli()

func key(b byte) ed25519.PrivateKey { return ed25519.NewKeyFromSeed(bytes.Repeat([]byte{b}, 32)) }

type builder struct {
	t    *testing.T
	list []*obs.Observation
}

func (b *builder) add(k ed25519.PrivateKey, mode obs.Mode, method string, dt time.Duration, subject []string, claims ...obs.Claim) {
	b.t.Helper()
	o, err := obs.Sign(obs.Body{Time: t0 + dt.Milliseconds(), Mode: mode, Method: method,
		Vantage: &obs.Vantage{Net: net1}, Subject: subject, Claims: claims}, k)
	if err != nil {
		b.t.Fatal(err)
	}
	b.list = append(b.list, o)
}

func one(t *testing.T, g *Graph, n int) {
	t.Helper()
	if len(g.Devices) != n {
		for _, d := range g.Devices {
			t.Logf("device %s %q idents=%v addrs=%v", d.ID, d.Label, d.Idents, d.Addrs)
		}
		for _, c := range g.Conflicts {
			t.Logf("conflict %+v", c)
		}
		t.Fatalf("got %d devices, want %d", len(g.Devices), n)
	}
}

// One NAS seen four ways by two observers, active and passive. Every piece
// of evidence must land on the same device.
func TestActivePassiveMultiObserverConverge(t *testing.T) {
	b := &builder{t: t}
	A, B := key(1), key(2)
	mac := "mac:00:11:32:aa:bb:cc"
	ip := "ip:" + net1 + "/192.168.1.20"
	// A, active: ARP sweep binds MAC↔IP; TCP probe binds IP↔SSH key.
	b.add(A, obs.ModeActive, "neigh", 0, []string{mac, ip})
	b.add(A, obs.ModeActive, "tcp", time.Minute, []string{ip, "ssh:SHA256:nas"}, obs.Claim{K: "port", V: "tcp/22"})
	// B, passive: DHCP request gives MAC + hostname; mDNS gives name ↔ IP.
	b.add(B, obs.ModePassive, "dhcp", -2*time.Hour, []string{mac, "host:" + net1 + "/nas"}, obs.Claim{K: "hostname", V: "nas"})
	b.add(B, obs.ModePassive, "mdns", 10*time.Minute, []string{"mdns:" + net1 + "/nas.local", ip},
		obs.Claim{K: "mdns.name", V: "nas.local"})
	b.add(B, obs.ModePassive, "mdns", 10*time.Minute, []string{"mdns:" + net1 + "/nas.local"},
		obs.Claim{K: "service", V: "mdns/_smb._tcp:445"})
	// A, active: ping (IP only) attaches via binding.
	b.add(A, obs.ModeActive, "ping", 5*time.Minute, []string{ip}, obs.Claim{K: "ttl", V: "64"})

	g := Build(b.list)
	one(t, g, 1)
	d := g.Devices[0]
	if d.Best("hostname") != "nas" || d.Best("ttl") != "64" || d.Best("service") == "" {
		t.Fatalf("claims not merged: %+v", d.Claims)
	}
	if len(d.Modes) != 2 || len(d.Observers) != 2 {
		t.Fatalf("modes=%v observers=%v", d.Modes, d.Observers)
	}
}

// The graph is a function of the set: any order gives the same result.
func TestOrderIndependent(t *testing.T) {
	b := &builder{t: t}
	A := key(1)
	for i, m := range []string{"mac:00:11:32:00:00:01", "mac:00:11:32:00:00:02", "mac:00:11:32:00:00:03"} {
		ip := "ip:" + net1 + "/10.0.0." + string(rune('1'+i))
		b.add(A, obs.ModeActive, "neigh", 0, []string{m, ip})
		b.add(A, obs.ModeActive, "ping", time.Second, []string{ip})
	}
	ref := Build(b.list)
	for range 20 {
		shuffled := append([]*obs.Observation(nil), b.list...)
		rand.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
		g := Build(shuffled)
		for i := range ref.Devices {
			if g.Devices[i].ID != ref.Devices[i].ID || g.Devices[i].Obs != ref.Devices[i].Obs {
				t.Fatal("result depends on input order")
			}
		}
	}
}

// DHCP gives the same IP to a different device a day later: two devices.
func TestIPReuseOverTime(t *testing.T) {
	b := &builder{t: t}
	ip := "ip:" + net1 + "/10.0.0.5"
	b.add(key(1), obs.ModeActive, "neigh", 0, []string{"mac:00:11:32:00:00:01", ip})
	b.add(key(1), obs.ModeActive, "neigh", 30*time.Hour, []string{"mac:00:11:32:00:00:02", ip})
	one(t, Build(b.list), 2)
}

// The same random MAC in two networks belongs to two (unrelated) devices.
func TestRandomMACScopedToNetwork(t *testing.T) {
	b := &builder{t: t}
	b.add(key(1), obs.ModeActive, "neigh", 0, []string{"lmac:nA/02:00:00:00:00:01", "ip:nA/10.0.0.2"})
	b.add(key(2), obs.ModeActive, "neigh", 0, []string{"lmac:nB/02:00:00:00:00:01", "ip:nB/10.0.0.2"})
	one(t, Build(b.list), 2)
}

// Two hosts both called "localhost" or reporting the same mDNS name but with
// different SSH keys must not be merged; the conflict is recorded.
func TestMediumMergeRefusedOnConflict(t *testing.T) {
	b := &builder{t: t}
	name := "host:" + net1 + "/server"
	b.add(key(1), obs.ModeActive, "x", 0, []string{"ssh:SHA256:one", name})
	b.add(key(1), obs.ModeActive, "x", 0, []string{"ssh:SHA256:two", name})
	g := Build(b.list)
	one(t, g, 2)
	if len(g.Conflicts) != 1 {
		t.Fatalf("want 1 conflict, got %v", g.Conflicts)
	}
}

// A router answering ARP for many addresses (proxy ARP / VRRP) must not
// swallow the hosts behind those addresses.
func TestProxyARPDoesNotSwallowHosts(t *testing.T) {
	b := &builder{t: t}
	gw := "lmac:" + net1 + "/00:00:5e:00:01:7a"
	for _, last := range []string{"1", "33", "34", "35"} {
		b.add(key(1), obs.ModeActive, "neigh", 0, []string{gw, "ip:" + net1 + "/10.82.0." + last})
	}
	b.add(key(1), obs.ModeActive, "tcp", time.Minute, []string{"ip:" + net1 + "/10.82.0.33", "ssh:SHA256:hidden"})
	g := Build(b.list)
	one(t, g, 2)
}

func TestLLDPLink(t *testing.T) {
	b := &builder{t: t}
	me := "mac:00:11:32:00:00:99"
	b.add(key(1), obs.ModeLocal, "self", 0, []string{"rkm:" + "aa", me})
	b.add(key(1), obs.ModePassive, "lldp", 0, []string{"lldp:mac/00:aa:bb:cc:dd:ee", "mac:00:aa:bb:cc:dd:ee"},
		obs.Claim{K: "link.l2", V: me}, obs.Claim{K: "link.port.a", V: "Gi1/0/7"}, obs.Claim{K: "link.port.b", V: "eth0"}, obs.Claim{K: "name", V: "core-sw"})
	g := Build(b.list)
	one(t, g, 2)
	if len(g.Links) != 1 || g.Links[0].Detail != "Gi1/0/7 ↔ eth0" {
		t.Fatalf("links: %+v", g.Links)
	}
}

// A UPnP device announces several UUIDs (root + embedded devices) from one IP.
func TestMultipleUPnPUUIDsOneDevice(t *testing.T) {
	b := &builder{t: t}
	ip := "ip:" + net1 + "/10.0.0.9"
	b.add(key(1), obs.ModePassive, "neigh", 0, []string{"mac:00:11:32:00:00:09", ip})
	b.add(key(2), obs.ModeActive, "ssdp", time.Minute, []string{ip, "uuid:aaaa"})
	b.add(key(2), obs.ModeActive, "ssdp", time.Minute, []string{ip, "uuid:bbbb"})
	g := Build(b.list)
	one(t, g, 1)
	if len(g.Conflicts) != 0 {
		t.Fatalf("unexpected conflicts %v", g.Conflicts)
	}
}
