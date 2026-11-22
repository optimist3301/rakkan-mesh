package resolve

import (
	"crypto/ed25519"
	"encoding/hex"
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

func hexKey(k ed25519.PrivateKey) string { return hex.EncodeToString(k.Public().(ed25519.PublicKey)) }

func info(g *Graph, k ed25519.PrivateKey) ObserverInfo {
	for _, i := range g.ObserverInfo {
		if i.Full == hexKey(k) {
			return i
		}
	}
	return ObserverInfo{}
}

func TestTrustWeightsAndBlocking(t *testing.T) {
	me, friend, stranger, troll := key(1), key(2), key(3), key(4)
	b := &builder{t: t}
	host := "mac:00:11:32:00:00:01"
	b.add(me, obs.ModeActive, "neigh", 0, []string{host}, obs.Claim{K: "name", V: "nas"})
	b.add(friend, obs.ModeActive, "x", 0, []string{host}, obs.Claim{K: "model", V: "DS920+"})
	b.add(stranger, obs.ModeActive, "x", 0, []string{host}, obs.Claim{K: "vendor", V: "Synology"})
	b.add(troll, obs.ModeActive, "x", 0, []string{host}, obs.Claim{K: "name", V: "pwned"})
	b.add(troll, obs.ModeActive, "x", 0, []string{"mac:00:11:32:66:66:66"}, obs.Claim{K: "name", V: "ghost"})

	book := trust.NewBook(trust.Entry{Key: hexKey(friend), Level: trust.Trusted, Name: "friend"},
		trust.Entry{Key: hexKey(troll), Level: trust.Blocked}).WithSelf(me.Public().(ed25519.PublicKey))
	g := BuildWith(b.list, Options{Trust: book})

	if len(g.Devices) != 1 {
		t.Fatalf("blocked observer's devices must vanish: %d devices", len(g.Devices))
	}
	d := g.Devices[0]
	if d.Best("name") != "nas" || len(d.Claims["name"]) != 1 {
		t.Fatalf("blocked claim leaked: %+v", d.Claims["name"])
	}
	if d.Claims["model"][0].Conf != 1000 || d.Claims["vendor"][0].Conf != 500 {
		t.Fatalf("trusted 1000 / unknown 500: model %d vendor %d", d.Claims["model"][0].Conf, d.Claims["vendor"][0].Conf)
	}
	if i := info(g, troll); i.Level != trust.Blocked || i.Ignored != 2 {
		t.Fatalf("troll info %+v", i)
	}
	if i := info(g, friend); i.Name != "friend" || i.Level != trust.Trusted {
		t.Fatalf("friend info %+v", i)
	}
	if info(g, me).Level != trust.Self {
		t.Fatal("own key must be self")
	}
}

// Identity poisoning: an unknown key claims the victim's MAC and an
// attacker's SSH key belong together. Trusted evidence says otherwise.
func TestUnknownCannotGlueIdentities(t *testing.T) {
	me, stranger := key(1), key(3)
	b := &builder{t: t}
	victim, attacker := "mac:00:11:32:00:00:01", "mac:00:11:32:00:00:66"
	b.add(me, obs.ModeActive, "x", 0, []string{victim, "ssh:SHA256:victim"})
	b.add(me, obs.ModeActive, "x", 0, []string{attacker, "ssh:SHA256:attacker"})
	b.add(stranger, obs.ModeActive, "x", time.Minute, []string{victim, "ssh:SHA256:attacker"})
	book := trust.NewBook().WithSelf(me.Public().(ed25519.PublicKey))

	g := BuildWith(b.list, Options{Trust: book})
	one(t, g, 2)
	if len(g.Conflicts) == 0 || g.Conflicts[0].Rule != "untrusted-identity" {
		t.Fatalf("poisoning must be refused and reported: %+v", g.Conflicts)
	}
	// The same assertion from a trusted key is believed (it is their word).
	book2 := trust.NewBook(trust.Entry{Key: hexKey(stranger), Level: trust.Trusted}).WithSelf(me.Public().(ed25519.PublicKey))
	if g := BuildWith(b.list, Options{Trust: book2}); len(g.Devices) != 1 {
		t.Fatalf("trusted assertion should merge: %d devices", len(g.Devices))
	}
	// An unknown key may still add identifiers that contradict nothing.
	b.add(stranger, obs.ModeActive, "x", 2*time.Minute, []string{victim, "uuid:new-thing"})
	g = BuildWith(b.list, Options{Trust: book})
	if d := byIdent(t, g, "uuid:new-thing"); !contains(d.Idents, victim) {
		t.Fatalf("non-conflicting addition refused: %v", d.Idents)
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func revoke(t *testing.T, b *builder, k ed25519.PrivateKey, dt, at time.Duration, next ed25519.PrivateKey) {
	t.Helper()
	claims := []obs.Claim{{K: "key.revoked", V: strconv.FormatInt(t0+at.Milliseconds(), 10)}}
	if next != nil {
		claims = append(claims, obs.Claim{K: "key.successor", V: "rkm:" + hexKey(next)})
	}
	b.add(k, obs.ModeLocal, "revoke", dt, []string{"rkm:" + hexKey(k)}, claims...)
}

func TestRevocationAndRotation(t *testing.T) {
	old, next, mallory := key(5), key(6), key(7)
	b := &builder{t: t}
	b.add(old, obs.ModeActive, "x", 0, []string{"mac:00:11:32:00:00:01"}, obs.Claim{K: "name", V: "before"})
	b.add(old, obs.ModeActive, "x", 2*time.Hour, []string{"mac:00:11:32:00:00:02"}, obs.Claim{K: "name", V: "after-compromise"})
	// The key was stolen at +1h; its owner revokes it at +3h, naming a successor.
	revoke(t, b, old, 3*time.Hour, time.Hour, next)
	b.add(next, obs.ModeActive, "x", 4*time.Hour, []string{"mac:00:11:32:00:00:03"}, obs.Claim{K: "name", V: "from-new-key"})
	// Someone else cannot revoke or replace a key they do not hold.
	b.add(mallory, obs.ModeLocal, "revoke", 4*time.Hour, []string{"rkm:" + hexKey(next)},
		obs.Claim{K: "key.revoked", V: strconv.FormatInt(t0, 10)})

	book := trust.NewBook(trust.Entry{Key: hexKey(old), Level: trust.Trusted, Name: "laptop"})
	g := BuildWith(b.list, Options{Trust: book})
	names := map[string]bool{}
	for _, d := range g.Devices {
		names[d.Best("name")] = true
	}
	if !names["before"] || names["after-compromise"] || !names["from-new-key"] {
		t.Fatalf("revocation window wrong: %v", names)
	}
	if i := info(g, old); i.RevokedAt != t0+time.Hour.Milliseconds() || i.Successor != hexKey(next) || i.Ignored != 1 {
		t.Fatalf("old key info %+v", i)
	}
	if i := info(g, next); i.Level != trust.Trusted || i.InheritedVia != hexKey(old) || i.RevokedAt != 0 {
		t.Fatalf("successor must inherit trust and not be revocable by others: %+v", i)
	}
	// An explicit decision about the new key wins over inheritance.
	book2 := trust.NewBook(trust.Entry{Key: hexKey(old), Level: trust.Trusted},
		trust.Entry{Key: hexKey(next), Level: trust.Blocked})
	if i := info(BuildWith(b.list, Options{Trust: book2}), next); i.Level != trust.Blocked {
		t.Fatalf("explicit level must win: %+v", i)
	}
}

func TestUnknownLocationWeighsLess(t *testing.T) {
	me, stranger := key(1), key(3)
	b := &builder{t: t}
	b.add(stranger, obs.ModeLocal, "manual", 0, []string{swMAC}, geo(50.0, 14.0, 10))
	g := BuildWith(b.list, Options{Trust: trust.NewBook().WithSelf(me.Public().(ed25519.PublicKey))})
	if c := byIdent(t, g, swMAC).Loc.Conf; math.Abs(c-0.5) > 1e-9 {
		t.Fatalf("unknown observer's position confidence %.3f, want 0.5", c)
	}
}

// A thief holding the key names their own successor: with two different
// successors on record, nobody inherits trust.
func TestAmbiguousSuccessorInheritsNothing(t *testing.T) {
	old, mine, thief := key(5), key(6), key(8)
	b := &builder{t: t}
	revoke(t, b, old, time.Hour, time.Hour, mine)
	revoke(t, b, old, 2*time.Hour, time.Hour, thief)
	b.add(mine, obs.ModeActive, "x", 3*time.Hour, []string{"mac:00:11:32:00:00:01"})
	b.add(thief, obs.ModeActive, "x", 3*time.Hour, []string{"mac:00:11:32:00:00:02"})
	g := BuildWith(b.list, Options{Trust: trust.NewBook(trust.Entry{Key: hexKey(old), Level: trust.Trusted})})
	if info(g, mine).Level != trust.Unknown || info(g, thief).Level != trust.Unknown || info(g, old).Successor != "" {
		t.Fatalf("mine %+v thief %+v", info(g, mine), info(g, thief))
	}
}
