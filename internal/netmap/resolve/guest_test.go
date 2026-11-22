package resolve

import (
	"crypto/ed25519"
	"fmt"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

func guestBook(me, member ed25519.PrivateKey, guests ...ed25519.PrivateKey) *trust.Book {
	es := []trust.Entry{{Key: hexKey(member), Level: trust.Trusted}}
	for _, g := range guests {
		es = append(es, trust.Entry{Key: hexKey(g), Level: trust.Unknown, Guest: true})
	}
	return trust.NewBook(es...).WithSelf(me.Public().(ed25519.PublicKey))
}

func mac(i int) string { return fmt.Sprintf("mac:00:11:32:00:00:%02x", i) }

// A guest starts below an unknown observer, rises as the members confirm
// what it says, and never reaches a member's weight.
func TestGuestEarnsWeight(t *testing.T) {
	me, member, guest := key(1), key(2), key(3)
	b := &builder{t: t}
	for i := 1; i <= 12; i++ {
		b.add(member, obs.ModeActive, "x", 0, []string{mac(i)}, obs.Claim{K: "hostname", V: fmt.Sprint("host", i)})
	}
	b.add(guest, obs.ModeActive, "x", 0, []string{mac(1)}, obs.Claim{K: "hostname", V: "host1"})
	book := guestBook(me, member, guest)

	g := BuildWith(b.list, Options{Trust: book})
	i := info(g, guest)
	if !i.Guest || i.Agree != 1 || i.Conflict != 0 || i.Weight < GuestWeight || i.Weight >= UnknownWeight {
		t.Fatalf("one confirmed fact: %+v", i)
	}
	for n := 2; n <= 12; n++ {
		b.add(guest, obs.ModeActive, "x", 0, []string{mac(n)}, obs.Claim{K: "hostname", V: fmt.Sprint("host", n)})
	}
	g = BuildWith(b.list, Options{Trust: book})
	if i := info(g, guest); i.Weight != UnknownWeight || i.Agree != 12 || i.Disputed {
		t.Fatalf("twelve confirmed facts: %+v", i)
	}
	if w := info(g, member).Weight; w != 1 {
		t.Fatalf("a member weighs %v", w)
	}
}

// A guest that contradicts the members loses weight, and past the limit
// vanishes from the map; the members' word stands either way.
func TestGuestContradictionsCostTrust(t *testing.T) {
	me, member, liar, honest := key(1), key(2), key(3), key(4)
	b := &builder{t: t}
	for i := 1; i <= 6; i++ {
		b.add(member, obs.ModeActive, "x", 0, []string{mac(i)}, obs.Claim{K: "hostname", V: fmt.Sprint("host", i)})
		b.add(honest, obs.ModeActive, "x", 0, []string{mac(i)}, obs.Claim{K: "hostname", V: fmt.Sprint("host", i)})
	}
	// The liar is right twice, wrong once: some weight lost, still heard.
	b.add(liar, obs.ModeActive, "x", 0, []string{mac(1)}, obs.Claim{K: "hostname", V: "host1"})
	b.add(liar, obs.ModeActive, "x", 0, []string{mac(2)}, obs.Claim{K: "hostname", V: "host2"})
	b.add(liar, obs.ModeActive, "x", 0, []string{mac(3)}, obs.Claim{K: "hostname", V: "evil"})
	book := guestBook(me, member, liar, honest)

	g := BuildWith(b.list, Options{Trust: book})
	l, h := info(g, liar), info(g, honest)
	if l.Agree != 2 || l.Conflict != 1 || l.Disputed || l.Weight >= h.Weight {
		t.Fatalf("liar %+v honest %+v", l, h)
	}
	var d3 *Device
	for _, d := range g.Devices {
		if d.Best("hostname") == "host3" {
			d3 = d
		}
	}
	if d3 == nil {
		t.Fatal("the members' name for the device is gone")
	}

	// More lies: now half of what it says is contradicted.
	b.add(liar, obs.ModeActive, "x", 0, []string{mac(4)}, obs.Claim{K: "hostname", V: "evil4"})
	g = BuildWith(b.list, Options{Trust: book})
	l = info(g, liar)
	if !l.Disputed || l.Ignored != 4 || l.Conflict != 2 {
		t.Fatalf("liar after more lies: %+v", l)
	}
	for _, d := range g.Devices {
		for _, v := range d.Claims["hostname"] {
			if v.Value == "evil" || v.Value == "evil4" {
				t.Fatalf("a disputed guest's claim is on the map: %+v", d.Claims)
			}
		}
	}
	if h := info(g, honest); h.Conflict != 0 || h.Disputed {
		t.Fatalf("the honest guest is hurt: %+v", h)
	}
}

// What a guest repeats scan after scan counts once.
func TestGuestFactsCountOnce(t *testing.T) {
	me, member, guest := key(1), key(2), key(3)
	b := &builder{t: t}
	b.add(member, obs.ModeActive, "x", 0, []string{mac(1)}, obs.Claim{K: "hostname", V: "nas"})
	for i := 0; i < 20; i++ {
		b.add(guest, obs.ModeActive, "x", 0, []string{mac(1)}, obs.Claim{K: "hostname", V: "wrong"})
	}
	g := BuildWith(b.list, Options{Trust: guestBook(me, member, guest)})
	if i := info(g, guest); i.Conflict != 1 {
		t.Fatalf("repeats counted: %+v", i)
	}
}

// The viewer's own decision outranks the guest mark and the cross-check.
func TestExplicitTrustOverridesGuest(t *testing.T) {
	me, member, guest := key(1), key(2), key(3)
	b := &builder{t: t}
	for i := 1; i <= 4; i++ {
		b.add(member, obs.ModeActive, "x", 0, []string{mac(i)}, obs.Claim{K: "hostname", V: "a"})
		b.add(guest, obs.ModeActive, "x", 0, []string{mac(i)}, obs.Claim{K: "hostname", V: "b"})
	}
	book := trust.NewBook(trust.Entry{Key: hexKey(member), Level: trust.Trusted}, trust.Entry{Key: hexKey(guest), Level: trust.Trusted}).
		WithSelf(me.Public().(ed25519.PublicKey))
	g := BuildWith(b.list, Options{Trust: book})
	if i := info(g, guest); i.Weight != 1 || i.Disputed || i.Guest {
		t.Fatalf("trusted guest: %+v", i)
	}
}

// A guest that tries to glue a member's device to another identity is
// counted as contradicted, as any unknown observer is.
func TestGuestIdentityGlueIsAContradiction(t *testing.T) {
	me, member, guest := key(1), key(2), key(3)
	b := &builder{t: t}
	b.add(member, obs.ModeActive, "x", 0, []string{mac(1), "ssh:SHA256:victim"})
	b.add(member, obs.ModeActive, "x", 0, []string{mac(66), "ssh:SHA256:attacker"})
	b.add(guest, obs.ModeActive, "x", 1, []string{mac(1), "ssh:SHA256:attacker"})
	g := BuildWith(b.list, Options{Trust: guestBook(me, member, guest)})
	if i := info(g, guest); i.Conflict != 1 {
		t.Fatalf("glue not counted: %+v", i)
	}
}
