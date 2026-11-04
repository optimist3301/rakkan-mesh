package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/mldht"
)

func fail(err error) int {
	fmt.Fprintln(os.Stderr, err)
	return 1
}

func idCmd(args []string) int {
	fs := flag.NewFlagSet("id", flag.ExitOnError)
	keyPath := fs.String("key", "node.key", "node private key, created if missing")
	fs.Parse(args)
	sk, err := membership.LoadOrCreateKey(*keyPath)
	if err != nil {
		return fail(err)
	}
	id, err := peer.IDFromPrivateKey(sk)
	if err != nil {
		return fail(err)
	}
	fmt.Println(id)
	return 0
}

func networkCmd(args []string) int {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		return 2
	}
	switch args[0] {
	case "init":
		return networkInit(args[1:])
	case "issue":
		return networkIssue(args[1:])
	case "revoke":
		return networkRevoke(args[1:])
	case "show":
		return networkShow(args[1:])
	case "open":
		return networkPolicy(args[1:], true)
	case "close":
		return networkPolicy(args[1:], false)
	case "announce":
		return networkAnnounce(args[1:])
	}
	fmt.Fprint(os.Stderr, usage)
	return 2
}

func networkInit(args []string) int {
	fs := flag.NewFlagSet("network init", flag.ExitOnError)
	out := fs.String("out", "root.key", "where to write the root key; keep it offline")
	fs.Parse(args)
	sk, n, err := membership.NewRootKey()
	if err != nil {
		return fail(err)
	}
	if err := membership.SaveKey(*out, sk); err != nil {
		return fail(err)
	}
	fmt.Fprintf(os.Stderr, "root key written to %s\nnetwork ID (give this to every node):\n", *out)
	fmt.Println(n.ID)
	return 0
}

func networkIssue(args []string) int {
	fs := flag.NewFlagSet("network issue", flag.ExitOnError)
	keyPath := fs.String("key", "root.key", "issuer key: the root key, or an admin node's key with -issuer-cert")
	issuerCert := fs.String("issuer-cert", "", "the admin's certificate, when the issuer is an admin")
	network := fs.String("network", "", "network ID (required with -issuer-cert)")
	subject := fs.String("peer", "", "peer ID of the node, from 'rakkan-node id'")
	roles := fs.String("roles", membership.RoleMember, "comma-separated roles: member, admin")
	days := fs.Int("days", 365, "validity in days")
	out := fs.String("out", "", "output file (default <peer>.cert)")
	fs.Parse(args)

	pid, err := peer.Decode(*subject)
	if err != nil {
		return fail(fmt.Errorf("-peer: %w", err))
	}
	sk, err := membership.LoadKey(*keyPath)
	if err != nil {
		return fail(err)
	}
	var parent *membership.SignedCert
	netID := *network
	if *issuerCert != "" {
		if parent, err = membership.LoadCert(*issuerCert); err != nil {
			return fail(err)
		}
		if netID == "" {
			return fail(errors.New("-network is required with -issuer-cert"))
		}
	} else {
		rootID, err := peer.IDFromPrivateKey(sk)
		if err != nil {
			return fail(err)
		}
		netID = rootID.String()
	}
	n, err := membership.ParseNetwork(netID)
	if err != nil {
		return fail(err)
	}
	now := time.Now()
	notAfter := now.Add(time.Duration(*days) * 24 * time.Hour)
	if parent != nil {
		pc, err := n.Verify(parent, now, nil)
		if err != nil {
			return fail(fmt.Errorf("issuer certificate: %w", err))
		}
		// A certificate may not outlive its issuer's.
		if notAfter.After(pc.NotAfter) {
			notAfter = pc.NotAfter
			fmt.Fprintf(os.Stderr, "validity shortened to the issuer's, until %s\n", notAfter.Format(time.DateOnly))
		}
	}
	sc, err := membership.Issue(sk, parent, membership.Cert{
		Network: n.ID, Subject: pid, Roles: strings.Split(*roles, ","),
		NotBefore: now.Add(-5 * time.Minute), NotAfter: notAfter,
	})
	if err != nil {
		return fail(err)
	}
	// Check it the way nodes will.
	c, err := n.Verify(sc, now, nil)
	if err != nil {
		return fail(fmt.Errorf("issued certificate does not verify: %w", err))
	}
	if *out == "" {
		*out = pid.String() + ".cert"
	}
	if err := os.WriteFile(*out, sc.Encode(), 0o644); err != nil {
		return fail(err)
	}
	fmt.Fprintf(os.Stderr, "certificate %s for %s written to %s\n", c.Serial, pid, *out)
	return 0
}

func networkRevoke(args []string) int {
	flags := flag.NewFlagSet("network revoke", flag.ExitOnError)
	keyPath := flags.String("key", "root.key", "root key")
	crlPath := flags.String("crl", "crl.json", "revocation list to extend; created if missing, rewritten in place")
	flags.Parse(args)
	serials := flags.Args()
	if len(serials) == 0 {
		return fail(errors.New("usage: rakkan-node network revoke [-key root.key] [-crl crl.json] <serial or peer ID>..."))
	}
	sk, err := membership.LoadKey(*keyPath)
	if err != nil {
		return fail(err)
	}
	rootID, err := peer.IDFromPrivateKey(sk)
	if err != nil {
		return fail(err)
	}
	n, err := membership.ParseNetwork(rootID.String())
	if err != nil {
		return fail(err)
	}
	next := membership.CRL{Network: n.ID, Seq: 1}
	if old, err := membership.LoadCRL(*crlPath); err == nil {
		c, err := n.VerifyCRL(old)
		if err != nil {
			return fail(err)
		}
		next.Seq, next.Revoked = c.Seq+1, c.Revoked
	} else if !errors.Is(err, fs.ErrNotExist) {
		return fail(err)
	}
	for _, s := range serials {
		if !slices.Contains(next.Revoked, s) {
			next.Revoked = append(next.Revoked, s)
		}
	}
	next.IssuedAt = time.Now()
	signed, err := membership.IssueCRL(sk, next)
	if err != nil {
		return fail(err)
	}
	if err := os.WriteFile(*crlPath, signed.Encode(), 0o644); err != nil {
		return fail(err)
	}
	fmt.Fprintf(os.Stderr, "revocation list %d (%d entries) written to %s\n", next.Seq, len(next.Revoked), *crlPath)
	return 0
}

// networkPolicy opens or closes the network to guests. The policy file is
// handed to a daemon with -policy, which spreads it.
func networkPolicy(args []string, open bool) int {
	name := "network close"
	if open {
		name = "network open"
	}
	flags := flag.NewFlagSet(name, flag.ExitOnError)
	keyPath := flags.String("key", "root.key", "root key")
	policyPath := flags.String("policy", "policy.json", "policy file; rewritten in place")
	pow := flags.Int("pow", 24, "proof of work a guest must show, in bits; each bit doubles it (24: about 6 CPU-seconds)")
	flags.Parse(args)
	sk, err := membership.LoadKey(*keyPath)
	if err != nil {
		return fail(err)
	}
	rootID, err := peer.IDFromPrivateKey(sk)
	if err != nil {
		return fail(err)
	}
	n, err := membership.ParseNetwork(rootID.String())
	if err != nil {
		return fail(err)
	}
	next := membership.Policy{Network: n.ID, Seq: 1, Open: open, IssuedAt: time.Now()}
	if open {
		next.PowBits = *pow
	}
	if old, err := membership.LoadPolicy(*policyPath); err == nil {
		p, err := n.VerifyPolicy(old)
		if err != nil {
			return fail(err)
		}
		next.Seq = p.Seq + 1
	} else if !errors.Is(err, fs.ErrNotExist) {
		return fail(err)
	}
	signed, err := membership.IssuePolicy(sk, next)
	if err != nil {
		return fail(err)
	}
	if err := os.WriteFile(*policyPath, signed.Encode(), 0o644); err != nil {
		return fail(err)
	}
	state := "closed"
	if open {
		state = fmt.Sprintf("open to guests proving %d bits of work", next.PowBits)
	}
	fmt.Fprintf(os.Stderr, "policy %d: network %s; written to %s\n", next.Seq, state, *policyPath)
	return 0
}

func networkShow(args []string) int {
	if len(args) != 1 {
		return fail(errors.New("usage: rakkan-node network show <file>"))
	}
	if sc, err := membership.LoadCert(args[0]); err == nil && sc.Body != nil {
		for depth := 0; sc != nil; depth++ {
			fmt.Printf("%s%s\n", strings.Repeat("  issued by: ", min(depth, 1)), sc.Body)
			sc = sc.Parent
		}
		return 0
	}
	b, err := os.ReadFile(args[0])
	if err != nil {
		return fail(err)
	}
	// A revocation list and a policy share a format; the body tells them
	// apart.
	var body struct {
		Body []byte `json:"body"`
	}
	if err := json.Unmarshal(b, &body); err != nil || body.Body == nil {
		return fail(fmt.Errorf("%s: not a certificate, revocation list or policy", args[0]))
	}
	fmt.Printf("%s\n", body.Body)
	return 0
}

// networkAnnounce signs the root's Mainline DHT record naming the
// announcers, members whose addresses newcomers look up there, and stores
// it. Members store it again from then on: run it again only to change
// the announcers.
func networkAnnounce(args []string) int {
	flags := flag.NewFlagSet("network announce", flag.ExitOnError)
	keyPath := flags.String("key", "root.key", "root key")
	list := flags.String("announcers", "", "comma-separated peer IDs of members with public addresses, running with -mainline")
	flags.Parse(args)
	sk, err := membership.LoadKey(*keyPath)
	if err != nil {
		return fail(err)
	}
	rootID, err := peer.IDFromPrivateKey(sk)
	if err != nil {
		return fail(err)
	}
	n, err := membership.ParseNetwork(rootID.String())
	if err != nil {
		return fail(err)
	}
	var ids []peer.ID
	for _, s := range strings.Split(*list, ",") {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		id, err := peer.Decode(s)
		if err != nil {
			return fail(fmt.Errorf("announcer %q: %w", s, err))
		}
		ids = append(ids, id)
	}
	item, err := mesh.SignBootstrapRecord(sk, n, ids, time.Now().Unix())
	if err != nil {
		return fail(err)
	}
	c, err := mldht.Listen(":0", mldht.DefaultBootstrap)
	if err != nil {
		return fail(err)
	}
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	stored, err := c.Put(ctx, item)
	if err != nil {
		return fail(err)
	}
	fmt.Fprintf(os.Stderr, "the record naming %d announcers is on %d DHT nodes\n", len(ids), stored)
	return 0
}
