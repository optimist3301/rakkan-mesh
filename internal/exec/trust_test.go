package exec

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	"github.com/optimist3301/rakkan-mesh/internal/trust"
	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// newOpenCluster is a cluster whose root admits guests proving 8 bits.
func newOpenCluster(t *testing.T) *cluster {
	c := newCluster(t)
	var err error
	c.policy, err = membership.IssuePolicy(c.root, membership.Policy{Network: c.net.ID, Seq: 1, IssuedAt: time.Now(), Open: true, PowBits: 8})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestGuestResultsAreCheckedAndCheatersBanned(t *testing.T) {
	c := newOpenCluster(t)
	// The requester checks every guest result, however reputable; the
	// guest has more free slots than the member executor, so it is picked
	// first.
	req := c.addNode(nodeSpec{trust: &trust.Config{MinVerify: 1}})
	member := c.addNode(nodeSpec{slots: 1, trust: &trust.Config{}})
	guest := c.addNode(nodeSpec{slots: 4, guest: true, trust: &trust.Config{}})
	settle(t, req, member, guest)

	in := []byte("honest work")
	res, err := req.svc.Run(t.Context(), Request{Module: w.Echo(), Input: in, Limits: limits()})
	if err != nil {
		t.Fatal(err)
	}
	if res.Node != guest.node.ID() || !res.Checked || !bytes.Equal(res.Output, in) {
		t.Fatalf("ran on %s, checked %v, output %q", res.Node.ShortString(), res.Checked, res.Output)
	}
	if len(res.Replicas) != 2 || res.Replicas[1].Node != member.node.ID() || !res.Replicas[0].Agreed || !res.Replicas[1].Agreed {
		t.Fatalf("replicas %+v", res.Replicas)
	}
	if sc := req.trust.Score(guest.node.ID()); sc <= 0 || sc > 0.1 {
		t.Fatalf("one verified result earned the guest a score of %f", sc)
	}

	// The guest starts lying: the member's run exposes it, the requester
	// returns the member's result, and the guest is banned.
	guest.svc.tamper = func(r *sandbox.Result) { r.Output = []byte("forged") }
	in = []byte("check me")
	res, err = req.svc.Run(t.Context(), Request{Module: w.Echo(), Input: in, Limits: limits()})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(res.Output, in) || res.Node != member.node.ID() || res.Replicas[0].Agreed {
		t.Fatalf("accepted %q from %s; replicas %+v", res.Output, res.Node.ShortString(), res.Replicas)
	}
	if !req.trust.Suspect(guest.node.ID()) || req.trust.Score(guest.node.ID()) != 0 {
		t.Fatal("the cheater kept its standing")
	}
	eventually(t, 5*time.Second, "cheater disconnected", func() bool { return !req.node.IsMember(guest.node.ID()) })
	// It gets no more work, even if it reconnects.
	for range 3 {
		res, err := req.svc.Run(t.Context(), Request{Module: w.Echo(), Input: in, Limits: limits()})
		if err != nil || res.Node == guest.node.ID() {
			t.Fatalf("ran on %v: %v", res, err)
		}
	}
}

func TestGuestErrorsAreCheckedToo(t *testing.T) {
	c := newOpenCluster(t)
	req := c.addNode(nodeSpec{trust: &trust.Config{}})
	member := c.addNode(nodeSpec{slots: 1, trust: &trust.Config{}})
	guest := c.addNode(nodeSpec{slots: 4, guest: true, trust: &trust.Config{}})
	settle(t, req, member, guest)
	// A deterministic failure from a guest is a claim like any other: the
	// task runs out of gas everywhere, so the guest is confirmed.
	lim := limits()
	lim.MaxGas = 10_000
	_, err := req.svc.Run(t.Context(), Request{Module: w.Spin(), Limits: lim})
	if !errors.Is(err, sandbox.ErrOutOfGas) {
		t.Fatalf("got %v", err)
	}
	if req.trust.Suspect(guest.node.ID()) || req.trust.Score(guest.node.ID()) == 0 {
		t.Fatal("an honest failure was not confirmed")
	}
}

func TestGuestCreditIsEarnedByComputing(t *testing.T) {
	// The guest's tasks need image kernels, which only the member offers,
	// so they cannot run on the guest itself. Their gas is deterministic:
	// measure it on a throwaway node.
	thumb := Request{Module: w.Thumbnail(64, 64, 0, 0), Input: gradientPNG(200, 100), Limits: bigLimits()}
	probe, err := newCluster(t).add(1).svc.Run(t.Context(), thumb)
	if err != nil {
		t.Fatal(err)
	}
	taskGas := probe.GasUsed
	thumb.Limits.MaxGas = taskGas

	c := newOpenCluster(t)
	// The member extends a guest credit for two such tasks, and none for
	// reputation, so the bilateral ledger alone decides.
	member := c.addNode(nodeSpec{slots: 1, trust: &trust.Config{FreeCredit: 2*taskGas + 1, CreditPerTrust: 1}})
	guest := c.addNode(nodeSpec{slots: 4, caps: []string{}, guest: true, trust: &trust.Config{}})
	settle(t, guest, member)
	settle(t, member, guest)

	ran := 0
	for ; ran < 10; ran++ {
		if _, err = guest.svc.Run(t.Context(), thumb); err != nil {
			break
		}
	}
	if ran != 2 || !strings.Contains(fmt.Sprint(err), "credit") {
		t.Fatalf("member ran %d tasks on credit for two, then: %v", ran, err)
	}
	debt := ledger(member, guest)

	// The guest computes for the member, paying down its debt, until the
	// member takes its tasks again.
	work := Request{Module: w.Countdown(2_000_000), Limits: limits()}
	for i := 0; ; i++ {
		res, err := member.svc.Run(t.Context(), work)
		if err != nil {
			t.Fatal(err)
		}
		if res.Node != guest.node.ID() {
			t.Fatalf("member's task ran on %s", res.Node.ShortString())
		}
		if d := ledger(member, guest); d >= debt {
			t.Fatalf("debt %d did not shrink from %d", d, debt)
		} else {
			debt = d
		}
		if _, err := guest.svc.Run(t.Context(), thumb); err == nil {
			t.Logf("credit restored after computing %d tasks for the member", i+1)
			break
		}
		if i == 20 {
			t.Fatal("guest still refused after computing for the member")
		}
	}
}

func eventually(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s", timeout, what)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// ledger returns what g owes n: the gas n spent for g beyond what g spent
// for n.
func ledger(n, g *testNode) int64 {
	for _, p := range n.trust.Peers() {
		if p.ID == g.node.ID() {
			return p.Served - p.Consumed
		}
	}
	panic(fmt.Sprintf("%s not in the ledger", g.node.ID()))
}

func TestGuestPaysForTasksThatFail(t *testing.T) {
	// A task that works and then traps costs its executor as much as one
	// that succeeds; were failures free, a guest could compute without
	// limit by trapping at the end of every task.
	work := w.CountdownTrap(2_000_000)
	r := sandbox.NewRunner(sandbox.DefaultLimits())
	defer r.Close(t.Context())
	_, err := r.Run(t.Context(), sandbox.Task{Module: work, Limits: limits()})
	taskGas, ok := sandbox.GasUsed(err)
	if !errors.Is(err, sandbox.ErrTrap) || !ok || taskGas < 10_000_000 {
		t.Fatalf("probe: %v, gas %d (%v)", err, taskGas, ok)
	}
	lim := limits()
	lim.MaxGas = taskGas + 1000

	c := newOpenCluster(t)
	// Credit for the gas of three such tasks, but the reservation of the
	// third pushes past it.
	member := c.addNode(nodeSpec{slots: 1, trust: &trust.Config{FreeCredit: 3 * taskGas, CreditPerTrust: 1}})
	guest := c.addNode(nodeSpec{guest: true, trust: &trust.Config{}})
	settle(t, guest, member)

	ran := 0
	for ; ran < 10; ran++ {
		_, err = guest.svc.Run(t.Context(), Request{Module: work, Limits: lim})
		if !errors.Is(err, sandbox.ErrTrap) {
			break
		}
	}
	if ran != 2 || !strings.Contains(fmt.Sprint(err), "credit") {
		t.Fatalf("member ran %d failing tasks on credit for two, then: %v", ran, err)
	}
	if d := ledger(member, guest); d != 2*taskGas {
		t.Fatalf("guest owes %d gas, want %d", d, 2*taskGas)
	}
}

func TestBilledGas(t *testing.T) {
	lim := limits()
	for _, tc := range []struct {
		name string
		res  *sandbox.Result
		err  error
		want int64
	}{
		{"success", &sandbox.Result{GasUsed: 42}, nil, 42},
		{"trap", nil, sandbox.WithGasUsed(fmt.Errorf("%w: x", sandbox.ErrTrap), 77), 77},
		{"out of gas", nil, sandbox.ErrOutOfGas, lim.MaxGas},
		{"timeout after running", nil, sandbox.WithGasUsed(sandbox.ErrTimeout, 5), lim.MaxGas},
		{"timeout killed", nil, sandbox.ErrTimeout, lim.MaxGas},
		{"worker oom", nil, isolate.ErrWorkerOOM, lim.MaxGas},
		{"invalid module", nil, sandbox.ErrInvalidModule, 0},
		{"worker crash", nil, isolate.ErrWorkerCrashed, 0},
	} {
		if got := billedGas(tc.res, tc.err, lim); got != tc.want {
			t.Errorf("%s: billed %d, want %d", tc.name, got, tc.want)
		}
	}
}
