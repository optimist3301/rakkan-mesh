package mesh

import (
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

func (tn *testNet) policy(t *testing.T, seq uint64, open bool, bits int) *membership.SignedPolicy {
	t.Helper()
	p, err := membership.IssuePolicy(tn.root, membership.Policy{Network: tn.net.ID, Seq: seq, IssuedAt: time.Now(), Open: open, PowBits: bits})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func withPolicy(p *membership.SignedPolicy) nodeOpt { return func(c *Config) { c.Policy = p } }

// guest starts a node without a certificate, proving bits of work.
func (tn *testNet) guest(t *testing.T, bits int, opts ...nodeOpt) *Node {
	t.Helper()
	key, id := newKey(t)
	pr, err := tn.net.Solve(t.Context(), id, bits)
	if err != nil {
		t.Fatal(err)
	}
	return start(t, tn.config(t, key, nil, append([]nodeOpt{func(c *Config) { c.Proof = &pr }}, opts...)...))
}

// guestWorking starts a node without a certificate whose proof does from
// lo to hi-1 bits of work. Solve(bits) takes the first nonce good enough,
// whose work is often more: once in sixteen, four bits more.
func (tn *testNet) guestWorking(t *testing.T, lo, hi int, opts ...nodeOpt) *Node {
	t.Helper()
	key, id := newKey(t)
	pr := membership.Proof{}
	for w := tn.net.Work(id, pr); w < lo || w >= hi; w = tn.net.Work(id, pr) {
		pr.Nonce++
	}
	return start(t, tn.config(t, key, nil, append([]nodeOpt{func(c *Config) { c.Proof = &pr }}, opts...)...))
}

func isGuest(n *Node, other *Node) bool {
	for _, m := range n.Members() {
		if m.ID == other.ID() {
			return m.Guest && m.Connected
		}
	}
	return false
}

func TestGuestsJoinOnlyAnOpenNetwork(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t)
	// Closed: no policy at all.
	g := tn.guest(t, 10, withBootstrap(a))
	time.Sleep(time.Second)
	if connectedMember(a, g.ID()) {
		t.Fatal("a closed network admitted a guest")
	}
	g.Close()

	// The root opens the network: the policy spreads from a member.
	if err := a.PublishPolicy(t.Context(), tn.policy(t, 1, true, 10)); err != nil {
		t.Fatal(err)
	}
	b := tn.member(t, withBootstrap(a))
	eventually(t, 5*time.Second, "policy reaches a new member", func() bool { return b.Policy().Open })

	weak := tn.guestWorking(t, 0, 10, withBootstrap(a))
	// Below 14 bits, so that raising the policy to 14 drops it below.
	g = tn.guestWorking(t, 10, 14, withBootstrap(a))
	eventually(t, 5*time.Second, "guest admitted", func() bool { return isGuest(a, g) && connectedMember(g, a.ID()) })
	if !a.IsMember(g.ID()) || a.IsCertified(g.ID()) || !a.IsCertified(b.ID()) {
		t.Fatal("guest and member are not told apart")
	}
	if got, err := echo(t.Context(), g, a.ID(), "hi"); err != nil || got != "hi\n" {
		t.Fatalf("guest echo: %q, %v", got, err)
	}
	// It finds the other member through the DHT, like a member would.
	eventually(t, 10*time.Second, "guest discovers the other member", func() bool { return isGuest(b, g) })
	// The member never counts it in. (The weak guest itself sees the
	// member, whose certificate is good, until the member hangs up.)
	for range 25 {
		if connectedMember(a, weak.ID()) {
			t.Fatal("a guest without enough work was admitted")
		}
		time.Sleep(20 * time.Millisecond)
	}

	// Raising the difficulty drops the guest, which solves a new proof and
	// comes back.
	solved := make(chan membership.Proof, 1)
	g.cfg.OnProof = func(pr membership.Proof) { solved <- pr }
	if err := a.PublishPolicy(t.Context(), tn.policy(t, 2, true, 14)); err != nil {
		t.Fatal(err)
	}
	select {
	case pr := <-solved:
		if tn.net.Work(g.ID(), pr) < 14 {
			t.Fatal("re-solved proof too weak")
		}
	case <-time.After(20 * time.Second):
		t.Fatal("guest did not re-solve its proof")
	}
	// Dropped guests are refused for denyAfterFailure; let it pass.
	a.mu.Lock()
	delete(a.denied, g.ID())
	a.mu.Unlock()
	eventually(t, 10*time.Second, "guest readmitted with the new proof", func() bool { return isGuest(a, g) })

	// Closing the network drops guests.
	if err := a.PublishPolicy(t.Context(), tn.policy(t, 3, false, 0)); err != nil {
		t.Fatal(err)
	}
	eventually(t, 5*time.Second, "guest dropped when the network closes", func() bool {
		return !connectedMember(a, g.ID()) && !connectedMember(b, g.ID())
	})
	if !connectedMember(a, b.ID()) {
		t.Fatal("closing the network dropped a member")
	}
}

func TestRevokedGuest(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t, withPolicy(tn.policy(t, 1, true, 8)))
	g := tn.guest(t, 8, withBootstrap(a))
	eventually(t, 5*time.Second, "guest admitted", func() bool { return isGuest(a, g) })
	crl, err := membership.IssueCRL(tn.root, membership.CRL{Network: tn.net.ID, Seq: 1, Revoked: []string{g.ID().String()}})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.PublishCRL(t.Context(), crl); err != nil {
		t.Fatal(err)
	}
	eventually(t, 5*time.Second, "revoked guest dropped", func() bool { return !connectedMember(a, g.ID()) })
}

func TestFetchPolicy(t *testing.T) {
	tn := newTestNet(t)
	a := tn.member(t)
	h, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if _, _, err := FetchPolicy(t.Context(), h, tn.net, a.AddrInfo()); err == nil {
		t.Fatal("a node without a policy served one")
	}
	a.PublishPolicy(t.Context(), tn.policy(t, 1, true, 12))
	_, p, err := FetchPolicy(t.Context(), h, tn.net, a.AddrInfo())
	if err != nil || !p.Open || p.PowBits != 12 {
		t.Fatalf("fetched %+v, %v", p, err)
	}
	other := newTestNet(t)
	if _, _, err := FetchPolicy(t.Context(), h, other.net, a.AddrInfo()); err == nil {
		t.Fatal("accepted another network's policy")
	}
}
