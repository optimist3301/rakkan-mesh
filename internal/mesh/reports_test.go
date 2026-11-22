package mesh

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/sealed"
)

func withReports(t *testing.T, sync time.Duration) nodeOpt {
	dir := t.TempDir()
	return func(c *Config) { c.Reports = ReportsConfig{Dir: dir, SyncInterval: sync} }
}

func rootPriv(t *testing.T, tn *testNet) ed25519.PrivateKey {
	t.Helper()
	raw, err := tn.root.Raw()
	if err != nil {
		t.Fatal(err)
	}
	return ed25519.PrivateKey(raw)
}

// A report travels the network sealed: members that carry it cannot read
// it, the root can.
func TestReportsReachOnlyTheRoot(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t, withReports(t, time.Hour))
	b := tn.member(t, withBootstrap(a), withReports(t, time.Hour))
	eventually(t, 10*time.Second, "a and b connected", func() bool { return connectedMember(a, b.ID()) })

	secret := []byte("10.0.0.0/24: printer, laptop")
	eventually(t, 20*time.Second, "gossip to b", func() bool {
		if _, err := a.PublishReport(context.Background(), append(bytes.Clone(secret), time.Now().String()...)); err != nil {
			t.Fatal(err)
		}
		time.Sleep(300 * time.Millisecond)
		return len(b.Reports(0, 10, 1<<20)) > 0
	})

	root := rootPriv(t, tn)
	ctx := ReportContext(tn.net.ID)
	for _, p := range b.Reports(0, 100, 1<<20) {
		if bytes.Contains(p.Data, []byte("printer")) {
			t.Fatal("the packet holds the plaintext")
		}
		pkt, c, err := VerifyReport(tn.net, p.Data, time.Now(), nil)
		if err != nil || c.Subject != a.ID() {
			t.Fatalf("verify: %v", err)
		}
		// Neither the relaying member nor any other key opens it.
		_, other, _ := ed25519.GenerateKey(nil)
		if _, err := sealed.Open(other, pkt.Sealed, ctx); err == nil {
			t.Fatal("another key opened a report")
		}
		got, err := sealed.Open(root, pkt.Sealed, ctx)
		if err != nil || !bytes.HasPrefix(got, secret) {
			t.Fatalf("root cannot read it: %v", err)
		}
	}
}

// A member that comes later, and an offline root, still get the history.
func TestReportsSyncToLateMember(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t, withReports(t, time.Hour))
	for i := range 3 {
		if _, err := a.PublishReport(context.Background(), []byte(fmt.Sprintf("report %d", i))); err != nil {
			t.Fatal(err)
		}
	}
	late := tn.member(t, withBootstrap(a), withReports(t, 200*time.Millisecond))
	eventually(t, 20*time.Second, "history pulled", func() bool { return len(late.Reports(0, 10, 1<<20)) == 3 })
}

func TestReportsRefusedWithoutCertOrSpool(t *testing.T) {
	tn := newTestNet(t)
	if _, err := tn.member(t).PublishReport(context.Background(), []byte("x")); err == nil {
		t.Fatal("a node without a spool sent a report")
	}
	a := tn.member(t, withReports(t, time.Hour))
	if _, err := a.PublishReport(context.Background(), nil); err == nil {
		t.Fatal("empty report")
	}
	if _, err := a.PublishReport(context.Background(), make([]byte, MaxReport+1)); err == nil {
		t.Fatal("oversized report")
	}
}

func TestVerifyReportRejects(t *testing.T) {
	tn := newTestNet(t)
	other := newTestNet(t)
	a := tn.member(t, withReports(t, time.Hour))
	if _, err := a.PublishReport(context.Background(), []byte("hello")); err != nil {
		t.Fatal(err)
	}
	data := a.Reports(0, 1, 1<<20)[0].Data
	if _, _, err := VerifyReport(tn.net, data, time.Now(), nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := VerifyReport(other.net, data, time.Now(), nil); err == nil {
		t.Error("a report of another network passed")
	}
	if _, _, err := VerifyReport(tn.net, data, time.Now().Add(2*time.Hour), nil); err == nil {
		t.Error("a report passed after its certificate expired")
	}
	if _, _, err := VerifyReport(tn.net, data, time.Now(), func(string) bool { return true }); err == nil {
		t.Error("a revoked member's report passed")
	}
	var p ReportPacket
	json.Unmarshal(data, &p)
	p.Sealed[len(p.Sealed)-1] ^= 1
	bad, _ := json.Marshal(p)
	if _, _, err := VerifyReport(tn.net, bad, time.Now(), nil); err == nil {
		t.Error("a tampered report passed")
	}
	// A member cannot pass a report off as another's: the certificate is its own.
	_, id := newKey(t)
	mallory := tn.issue(t, id, []string{membership.RoleMember}, nil, nil)
	p2 := p
	p2.Cert = mallory
	p2.Sealed[len(p2.Sealed)-1] ^= 1 // back to the original
	forged, _ := json.Marshal(p2)
	if _, _, err := VerifyReport(tn.net, forged, time.Now(), nil); err == nil {
		t.Error("a report under someone else's certificate passed")
	}
}

func TestSpoolEvictsOldestAndSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	s, err := openSpool(dir, 1<<20, 3, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for i := range 5 {
		id, added, err := s.Put([]byte(fmt.Sprintf("packet %d", i)))
		if err != nil || !added {
			t.Fatal(err, added)
		}
		ids = append(ids, id)
	}
	if _, added, _ := s.Put([]byte("packet 4")); added {
		t.Fatal("a duplicate was added")
	}
	if got := s.IDs(10); len(got) != 3 || got[0] != ids[2] {
		t.Fatalf("kept %v", got)
	}
	s2, err := openSpool(dir, 1<<20, 3, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	list := s2.After(0, 10, 1<<20, nil)
	if len(list) != 3 || string(list[0].Data) != "packet 2" || string(list[2].Data) != "packet 4" {
		t.Fatalf("after restart: %d packets", len(list))
	}
	if next := s2.After(list[0].Seq, 10, 1<<20, nil); len(next) != 2 {
		t.Fatalf("cursor: %d", len(next))
	}
	if _, _, err := s2.Put([]byte("packet 5")); err != nil {
		t.Fatal(err)
	}
	if list2 := s2.After(list[2].Seq, 10, 1<<20, nil); len(list2) != 1 {
		t.Fatalf("new after cursor: %d", len(list2))
	}
}

// A guest's report reaches members and the root, marked as a guest's; the
// plain VerifyReport still refuses it, and a closed network takes none.
func TestGuestReports(t *testing.T) {
	tn := newTestNet(t)
	pol := withPolicy(tn.policy(t, 1, true, 8))
	a := tn.member(t, pol, withReports(t, 200*time.Millisecond))
	g := tn.guest(t, 8, pol, withBootstrap(a), withReports(t, time.Hour))
	eventually(t, 15*time.Second, "guest connected", func() bool { return isGuest(a, g) })

	eventually(t, 25*time.Second, "the member holds the guest's report", func() bool {
		if _, err := g.PublishReport(context.Background(), []byte("guest says "+time.Now().String())); err != nil {
			t.Fatal(err)
		}
		time.Sleep(300 * time.Millisecond)
		return len(a.Reports(0, 10, 1<<20)) > 0
	})
	data := a.Reports(0, 1, 1<<20)[0].Data
	if _, _, err := VerifyReport(tn.net, data, time.Now(), nil); err == nil {
		t.Fatal("the member-only check took a guest's report")
	}
	if _, _, err := VerifyReportGuests(tn.net, data, time.Now(), nil, 30); err == nil {
		t.Fatal("a proof below the policy's work was taken")
	}
	pkt, c, err := VerifyReportGuests(tn.net, data, time.Now(), nil, 8)
	if err != nil || c.Subject != g.ID() || !c.HasRole(membership.RoleGuest) {
		t.Fatalf("verify: %v %v", c, err)
	}
	got, err := sealed.Open(rootPriv(t, tn), pkt.Sealed, ReportContext(tn.net.ID))
	if err != nil || !bytes.HasPrefix(got, []byte("guest says")) {
		t.Fatalf("root cannot read it: %v", err)
	}
	// Someone else's signature on a guest's packet does not hold.
	var forged ReportPacket
	json.Unmarshal(data, &forged)
	forged.Sealed = append(bytes.Clone(forged.Sealed[:len(forged.Sealed)-1]), forged.Sealed[len(forged.Sealed)-1]^1)
	bad, _ := json.Marshal(forged)
	if _, _, err := VerifyReportGuests(tn.net, bad, time.Now(), nil, 8); err == nil {
		t.Fatal("a tampered guest report verified")
	}
}

// A guest without a proof cannot send; the pulling member reaches a guest's
// report even when gossip missed it.
func TestGuestReportPulledByMember(t *testing.T) {
	tn := newTestNet(t)
	pol := withPolicy(tn.policy(t, 1, true, 8))
	a := tn.member(t, pol, withReports(t, 200*time.Millisecond))
	g := tn.guest(t, 8, pol, withBootstrap(a), withReports(t, time.Hour))
	eventually(t, 15*time.Second, "guest connected", func() bool { return isGuest(a, g) })
	// Put a packet in the guest's spool without gossiping it.
	id, err := g.PublishReport(context.Background(), []byte("quiet"))
	if err != nil || id == "" {
		t.Fatal(err)
	}
	eventually(t, 25*time.Second, "the member has it", func() bool { return len(a.Reports(0, 10, 1<<20)) > 0 })
}
