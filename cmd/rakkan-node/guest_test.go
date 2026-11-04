package main

import (
	"crypto/rand"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/mldht"
)

// A guest with no -bootstrap and no -policy takes the policy from an
// announcer the Mainline DHT names.
func TestFetchPolicyThroughMainline(t *testing.T) {
	root, n, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	d, err := mldht.NewTestNetwork(20)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	key, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	id, _ := peer.IDFromPrivateKey(key)
	cert, err := membership.Issue(root, nil, membership.Cert{Network: n.ID, Subject: id, Roles: []string{membership.RoleMember},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour)})
	if err != nil {
		t.Fatal(err)
	}
	pol, err := membership.IssuePolicy(root, membership.Policy{Network: n.ID, Seq: 1, IssuedAt: time.Now(), Open: true, PowBits: 6})
	if err != nil {
		t.Fatal(err)
	}
	node, err := mesh.New(mesh.Config{Key: key, Network: n, Cert: cert, Policy: pol,
		ListenAddrs: []string{"/ip4/127.0.0.1/tcp/0"}, Mainline: true, MainlineBootstrap: d.Bootstrap(), MainlineInterval: 200 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	defer node.Close()
	item, err := mesh.SignBootstrapRecord(root, n, []peer.ID{id}, 1)
	if err != nil {
		t.Fatal(err)
	}
	c, err := mldht.Listen("127.0.0.1:0", d.Bootstrap())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Put(t.Context(), item); err != nil {
		t.Fatal(err)
	}
	if _, _, err := fetchPolicyFrom(t.Context(), n, nil, false, d.Bootstrap()); err == nil {
		t.Error("a policy without -bootstrap or -mainline")
	}
	var got *membership.Policy
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, got, err = fetchPolicyFrom(t.Context(), n, nil, true, d.Bootstrap()); err == nil {
			break
		}
		time.Sleep(200 * time.Millisecond) // the announcer's first round
	}
	if err != nil || !got.Open || got.PowBits != 6 {
		t.Fatalf("policy %+v, %v", got, err)
	}
}
