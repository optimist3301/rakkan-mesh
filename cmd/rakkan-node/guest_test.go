package main

import (
	"crypto/rand"
	"net"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/mldht"
	"github.com/optimist3301/rakkan-mesh/internal/ygg"
)

// A guest with no -bootstrap and no -policy takes the policy from an
// announcer the Mainline DHT names.
func TestFetchPolicyThroughMainline(t *testing.T) { fetchPolicyThroughMainline(t, false) }

// The same with an announcer reachable only over Yggdrasil: it listens on
// no IP address.
func TestFetchPolicyOverYggdrasil(t *testing.T) { fetchPolicyThroughMainline(t, true) }

func fetchPolicyThroughMainline(t *testing.T, overYgg bool) {
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
	cfg := mesh.Config{Key: key, Network: n, Cert: cert, Policy: pol,
		ListenAddrs: []string{"/ip4/127.0.0.1/tcp/0"}, Mainline: true, MainlineBootstrap: d.Bootstrap(), MainlineInterval: 200 * time.Millisecond}
	src := policySource{mainline: true, dht: d.Bootstrap()}
	if overYgg {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		uri := "tcp://" + l.Addr().String()
		l.Close()
		cfg.ListenAddrs = []string{}
		cfg.Yggdrasil, cfg.YggdrasilPeers, cfg.YggdrasilListen = true, []string{}, []string{uri}
		src.ygg = &ygg.Config{Peers: []string{uri}}
	}
	node, err := mesh.New(cfg)
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
	if _, _, err := fetchPolicy(t.Context(), n, policySource{dht: d.Bootstrap()}); err == nil {
		t.Error("a policy without -bootstrap or -mainline")
	}
	var got *membership.Policy
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, got, err = fetchPolicy(t.Context(), n, src); err == nil {
			break
		}
		time.Sleep(200 * time.Millisecond) // the announcer's first round
	}
	if err != nil || !got.Open || got.PowBits != 6 {
		t.Fatalf("policy %+v, %v", got, err)
	}
}
