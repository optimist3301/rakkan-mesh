package obs

import (
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"flag"
	"net"
	"net/netip"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite spec test vectors")

func testKey() ed25519.PrivateKey {
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}
	return ed25519.NewKeyFromSeed(seed)
}

func sample() Body {
	return Body{
		Time:    1767225600000, // 2026-01-01T00:00:00Z
		Mode:    ModePassive,
		Method:  "dhcp",
		Vantage: &Vantage{Iface: "eth0", Net: "nexample"},
		Subject: []string{"mac:00:11:22:33:44:55", "host:nexample/nas", "mac:00:11:22:33:44:55"},
		Claims: []Claim{
			{K: "dhcp.vendor", V: "MSFT 5.0"},
			{K: "hostname", V: "nas", C: 1000},
			{K: "dhcp.fp", V: "1,3,6,15", C: 900},
		},
	}
}

func TestDeterministicAcrossOrdering(t *testing.T) {
	k := testKey()
	a, err := Sign(sample(), k)
	if err != nil {
		t.Fatal(err)
	}
	b := sample()
	b.Subject = []string{"host:nexample/nas", "mac:00:11:22:33:44:55"}
	b.Claims[0], b.Claims[2] = b.Claims[2], b.Claims[0]
	b.Claims[1].C = 0 // same as 1000
	o2, err := Sign(b, k)
	if err != nil {
		t.Fatal(err)
	}
	if a.ID != o2.ID {
		t.Fatalf("ids differ for equivalent content: %x vs %x", a.ID, o2.ID)
	}
	if len(a.Subject) != 2 {
		t.Fatalf("subject not deduped: %v", a.Subject)
	}
}

func TestDecodeVerifies(t *testing.T) {
	o, err := Sign(sample(), testKey())
	if err != nil {
		t.Fatal(err)
	}
	back, err := Decode(o.Raw, o.Sig)
	if err != nil {
		t.Fatal(err)
	}
	if back.ID != o.ID || back.Method != "dhcp" || len(back.Claims) != 3 {
		t.Fatalf("roundtrip mismatch: %+v", back.Body)
	}
	bad := append([]byte(nil), o.Raw...)
	bad[len(bad)-1] ^= 1
	if _, err := Decode(bad, o.Sig); err != ErrBadSignature {
		t.Fatalf("tampered body: got %v, want bad signature", err)
	}
}

func TestRejectsNonCanonical(t *testing.T) {
	k := testKey()
	b := sample()
	b.V, b.Observer = SchemaVersion, k.Public().(ed25519.PublicKey)
	// Deliberately unsorted subject, encoded without canonicalization.
	b.Subject = []string{"mac:00:11:22:33:44:55", "host:nexample/nas"}
	raw, err := Marshal(&b)
	if err != nil {
		t.Fatal(err)
	}
	id := ComputeID(raw)
	sig := ed25519.Sign(k, sigMessage(id))
	if _, err := Decode(raw, sig); err != ErrNotCanonical {
		t.Fatalf("got %v, want ErrNotCanonical", err)
	}
}

func TestFutureVersionKeptVerbatim(t *testing.T) {
	k := testKey()
	raw, _ := Marshal(map[int]any{0: 2, 1: []byte(k.Public().(ed25519.PublicKey)), 42: "new field"})
	id := ComputeID(raw)
	o, err := Decode(raw, ed25519.Sign(k, sigMessage(id)))
	if err != nil {
		t.Fatal(err)
	}
	if o.V != 2 || string(o.Raw) != string(raw) {
		t.Fatal("future-version body not preserved")
	}
}

func TestIdentifiers(t *testing.T) {
	gw, _ := net.ParseMAC("00:00:5e:00:01:7a")
	gwIP := netip.MustParseAddr("10.82.0.1")
	p := netip.MustParsePrefix("10.82.152.58/16")
	n1 := NetID(gw, gwIP, p)
	if n1 != NetID(gw, gwIP, netip.MustParsePrefix("10.82.0.0/16")) {
		t.Fatal("netid must depend on masked prefix only")
	}
	if got := MACID(gw, n1); got != "lmac:"+n1+"/00:00:5e:00:01:7a" {
		t.Fatalf("VRRP mac must be scoped, got %s", got)
	}
	rnd, _ := net.ParseMAC("b2:a5:14:8a:46:70")
	if Kind(MACID(rnd, n1)) != "lmac" {
		t.Fatal("locally administered mac must be scoped")
	}
	g, _ := net.ParseMAC("1C:CE:51:E6:C8:3D")
	if MACID(g, n1) != "mac:1c:ce:51:e6:c8:3d" {
		t.Fatal("global mac must be unscoped, lowercase")
	}
	id := IPID(n1, netip.MustParseAddr("fe80::1%wlan0"))
	if Scope(id) != n1 || Value(id) != "fe80::1" {
		t.Fatalf("bad ip id %s", id)
	}
	if NameID("host", n1, "localhost") != "" {
		t.Fatal("localhost must be blocked")
	}
}

// TestVector pins the wire format (spec §8). Run with -update to regenerate
// after an intentional format change.
func TestVector(t *testing.T) {
	o, err := Sign(sample(), testKey())
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{
		"seed":       hex.EncodeToString(testKey().Seed()),
		"body_bytes": hex.EncodeToString(o.Raw),
		"id":         hex.EncodeToString(o.ID[:]),
		"sig":        hex.EncodeToString(o.Sig),
	}
	path := filepath.Join("..", "..", "..", "docs", "netmap-spec", "vectors", "obs-dhcp.json")
	if *update {
		os.MkdirAll(filepath.Dir(path), 0o755)
		b, _ := json.MarshalIndent(got, "", "  ")
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test -update once)", err)
	}
	var want map[string]string
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s: got %s, want %s", k, got[k], v)
		}
	}
}

func TestGeoClaim(t *testing.T) {
	v := FormatGeo(50.087, 14.42, 12.34)
	lat, lon, r, err := ParseGeo(v)
	if err != nil || lat != 50.087 || lon != 14.42 || r != 12.3 {
		t.Fatalf("%q → %v %v %v %v", v, lat, lon, r, err)
	}
	for _, bad := range []string{"", "1,2", "91,0,1", "0,181,1", "0,0,0", "0,0,-1", "0,0,3e9", "a,b,c", "NaN,0,5", "Inf,0,5"} {
		if _, _, _, err := ParseGeo(bad); err == nil {
			t.Errorf("%q must be rejected", bad)
		}
	}
}

// A VRRP failover changes the gateway's virtual MAC; the network must keep
// its identifier. Networks with a real gateway MAC keep the v1 value.
func TestNetIDVirtualGateway(t *testing.T) {
	p := netip.MustParsePrefix("10.82.0.0/16")
	gwIP := netip.MustParseAddr("10.82.0.1")
	var ids []string
	for _, m := range []string{"00:00:5e:00:01:7a", "00:00:5e:00:01:70", "00:00:0c:07:ac:01", "00:00:0c:9f:f0:02", "00:07:b4:00:01:02"} {
		mac, _ := net.ParseMAC(m)
		ids = append(ids, NetID(mac, gwIP, p))
	}
	for _, id := range ids[1:] {
		if id != ids[0] {
			t.Fatalf("virtual gateway MACs must give one network id: %v", ids)
		}
	}
	// Same virtual MAC, different gateway address or prefix: different networks.
	vrrp, _ := net.ParseMAC("00:00:5e:00:01:01")
	a := NetID(vrrp, netip.MustParseAddr("192.168.10.1"), netip.MustParsePrefix("192.168.10.0/24"))
	b := NetID(vrrp, netip.MustParseAddr("192.168.20.1"), netip.MustParsePrefix("192.168.20.0/24"))
	c := NetID(vrrp, netip.MustParseAddr("192.168.10.254"), netip.MustParsePrefix("192.168.10.0/24"))
	if a == b || a == c {
		t.Fatal("different gateways must give different ids")
	}
	// Real gateway MAC: unchanged from v1 (maps already collected must still merge),
	// and independent of the gateway IP.
	real, _ := net.ParseMAC("00:11:32:aa:bb:cc")
	rp := netip.MustParsePrefix("192.168.1.0/24")
	if got := NetID(real, netip.MustParseAddr("192.168.1.1"), rp); got != "nt6wc7mvnxqytfh72" {
		t.Fatalf("real-MAC network id changed: %s", got)
	}
	if NetID(real, netip.Addr{}, rp) != "nt6wc7mvnxqytfh72" {
		t.Fatal("real-MAC id must not depend on the gateway IP")
	}
	// Virtual MAC but unknown gateway IP: falls back to the MAC (never empty).
	if NetID(vrrp, netip.Addr{}, rp) == "" {
		t.Fatal("empty id")
	}
}
