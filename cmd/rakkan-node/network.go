package main

import (
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
		return fail(errors.New("usage: rakkan-node network revoke [-key root.key] [-crl crl.json] <serial>..."))
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
	fmt.Fprintf(os.Stderr, "revocation list %d (%d serials) written to %s\n", next.Seq, len(next.Revoked), *crlPath)
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
	crl, err := membership.LoadCRL(args[0])
	if err != nil {
		return fail(err)
	}
	fmt.Printf("%s\n", crl.Body)
	return 0
}
