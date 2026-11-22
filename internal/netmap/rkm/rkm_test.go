package rkm

import (
	"bytes"
	"crypto/ed25519"
	"fmt"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

func key(b byte) ed25519.PrivateKey {
	seed := bytes.Repeat([]byte{b}, 32)
	return ed25519.NewKeyFromSeed(seed)
}

func makeObs(t *testing.T, k ed25519.PrivateKey, mode obs.Mode, n int) []*obs.Observation {
	t.Helper()
	var out []*obs.Observation
	for i := range n {
		o, err := obs.Sign(obs.Body{
			Time: int64(1767225600000 + i), Mode: mode, Method: "neigh",
			Subject: []string{fmt.Sprintf("mac:00:00:00:00:00:%02x", i+1), fmt.Sprintf("ip:nx/10.0.0.%d", i+1)},
		}, k)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, o)
	}
	return out
}

func roundtrip(t *testing.T, list []*obs.Observation) *Result {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, list, "test"); err != nil {
		t.Fatal(err)
	}
	r, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !r.DigestOK || len(r.Rejected) > 0 {
		t.Fatalf("digest ok=%v rejected=%v", r.DigestOK, r.Rejected)
	}
	return r
}

func digestOf(list []*obs.Observation) [32]byte {
	ids := make([][32]byte, len(list))
	for i, o := range list {
		ids[i] = o.ID
	}
	return SetDigest(ids)
}

// Active and passive observations from different observers merge as a plain
// set union: commutative, associative, idempotent.
func TestMergeIsCRDT(t *testing.T) {
	a := makeObs(t, key(1), obs.ModeActive, 5)
	b := makeObs(t, key(2), obs.ModePassive, 7)
	c := append(makeObs(t, key(3), obs.ModeLocal, 3), a[0], b[0]) // overlap

	cat := func(ls ...[]*obs.Observation) []*obs.Observation {
		var out []*obs.Observation
		for _, l := range ls {
			out = append(out, l...)
		}
		return roundtrip(t, out).Observations
	}
	ab := cat(a, b)
	ba := cat(b, a)
	if digestOf(ab) != digestOf(ba) {
		t.Fatal("merge not commutative")
	}
	if digestOf(cat(ab, c)) != digestOf(cat(a, cat(b, c))) {
		t.Fatal("merge not associative")
	}
	if digestOf(cat(ab, ab)) != digestOf(ab) || len(cat(ab, ab)) != 12 {
		t.Fatal("merge not idempotent")
	}
	if len(cat(a, b, c)) != 15 {
		t.Fatalf("overlap not deduplicated")
	}
}

func TestDeterministicFileContent(t *testing.T) {
	a := makeObs(t, key(1), obs.ModeActive, 4)
	r1 := roundtrip(t, a)
	rev := []*obs.Observation{a[3], a[2], a[1], a[0]}
	r2 := roundtrip(t, rev)
	for i := range r1.Observations {
		if r1.Observations[i].ID != r2.Observations[i].ID {
			t.Fatal("records not sorted by id")
		}
	}
}

func TestBadRecordRejectedNotFatal(t *testing.T) {
	a := makeObs(t, key(1), obs.ModeActive, 3)
	forged := *a[1]
	forged.Sig = bytes.Repeat([]byte{0}, 64)
	var buf bytes.Buffer
	if err := Write(&buf, []*obs.Observation{a[0], &forged, a[2]}, "t"); err != nil {
		t.Fatal(err)
	}
	r, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Observations) != 2 || len(r.Rejected) != 1 || r.DigestOK {
		t.Fatalf("got %d ok, %d rejected, digestOK=%v", len(r.Observations), len(r.Rejected), r.DigestOK)
	}
}
