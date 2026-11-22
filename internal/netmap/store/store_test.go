package store

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

func TestAddIsIdempotent(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "m.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	k := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{1}, 32))
	var list []*obs.Observation
	for i, mode := range []obs.Mode{obs.ModeActive, obs.ModePassive, obs.ModeLocal} {
		o, err := obs.Sign(obs.Body{Time: int64(i + 1), Mode: mode, Method: "neigh", Subject: []string{"mac:00:11:22:33:44:55"}}, k)
		if err != nil {
			t.Fatal(err)
		}
		list = append(list, o)
	}
	if n, _ := st.Add(list); n != 3 {
		t.Fatalf("first add: %d new", n)
	}
	if n, _ := st.Add(append(list, list...)); n != 0 {
		t.Fatalf("re-add: %d new, want 0", n)
	}
	all, err := st.All()
	if err != nil || len(all) != 3 {
		t.Fatalf("all: %d, %v", len(all), err)
	}
	s, _ := st.Stats()
	if s.ByMode[obs.ModePassive] != 1 || s.Observers != 1 {
		t.Fatalf("stats %+v", s)
	}
}

func TestVerifyFindsTampering(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "m.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	k := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{1}, 32))
	o, _ := obs.Sign(obs.Body{Time: 1, Mode: obs.ModeActive, Method: "neigh", Subject: []string{"mac:00:11:22:33:44:55"}}, k)
	st.Add([]*obs.Observation{o})
	if ok, bad, _ := st.Verify(); ok != 1 || bad != 0 {
		t.Fatalf("clean store: ok=%d bad=%d", ok, bad)
	}
	st.db.Exec(`UPDATE obs SET sig = zeroblob(64)`)
	if ok, bad, _ := st.Verify(); ok != 0 || bad != 1 {
		t.Fatalf("tampered store: ok=%d bad=%d", ok, bad)
	}
}

func TestUnsentPruneAndWindow(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "m.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	_, mine, _ := ed25519.GenerateKey(rand.Reader)
	_, other, _ := ed25519.GenerateKey(rand.Reader)
	mk := func(k ed25519.PrivateKey, ip string, ago time.Duration) *obs.Observation {
		o, err := obs.Sign(obs.Body{Time: time.Now().Add(-ago).UnixMilli(), Mode: obs.ModeActive, Method: "ping", Subject: []string{"ip:" + ip}}, k)
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	oldSent, oldUnsent, fresh, foreign := mk(mine, "10.0.0.1", 48*time.Hour), mk(mine, "10.0.0.2", 47*time.Hour), mk(mine, "10.0.0.3", time.Minute), mk(other, "10.0.0.4", 49*time.Hour)
	if _, err := st.Add([]*obs.Observation{oldSent, oldUnsent, fresh, foreign}); err != nil {
		t.Fatal(err)
	}
	me := [][]byte{mine.Public().(ed25519.PublicKey)}
	if got, _ := st.Unsent(me, 10); len(got) != 3 {
		t.Fatalf("unsent: %d", len(got))
	}
	if got, _ := st.Unsent(me, 2); len(got) != 2 {
		t.Fatalf("limit: %d", len(got))
	}
	if err := st.MarkSent([]*obs.Observation{oldSent, fresh}); err != nil {
		t.Fatal(err)
	}
	if got, _ := st.Unsent(me, 10); len(got) != 1 || got[0].ID != oldUnsent.ID {
		t.Fatalf("after marking: %v", got)
	}
	// Old observations go, except own ones that never left.
	n, err := st.Prune(time.Now().Add(-24*time.Hour).UnixMilli(), me)
	if err != nil || n != 2 {
		t.Fatalf("prune: %d %v", n, err)
	}
	all, _ := st.All()
	if len(all) != 2 {
		t.Fatalf("left %d", len(all))
	}
	win, _ := st.Since(time.Now().Add(-time.Hour).UnixMilli(), 0)
	if len(win) != 1 || win[0].ID != fresh.ID {
		t.Fatalf("window: %d", len(win))
	}
	if win, _ = st.Since(0, 1); len(win) != 1 || win[0].ID != fresh.ID {
		t.Fatal("the newest one is kept by the limit")
	}
	v1, _ := st.Version()
	st.Add([]*obs.Observation{mk(mine, "10.0.0.9", 0)})
	if v2, _ := st.Version(); v1 == v2 {
		t.Fatal("version did not change")
	}
}
