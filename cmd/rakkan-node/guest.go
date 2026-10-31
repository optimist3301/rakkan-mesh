package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
)

// guestProof returns the proof of work a node without a certificate joins
// with: the saved one if it satisfies the policy, or a new one, solved and
// saved. The policy comes from known (the -policy file), or else from a
// bootstrap node, which serves it to anyone: the root's signature makes it
// safe to take from strangers.
func guestProof(ctx context.Context, n *membership.Network, id peer.ID, known *membership.SignedPolicy, bootstrap []peer.AddrInfo, path string) (membership.Proof, *membership.SignedPolicy, error) {
	signed, pol := known, (*membership.Policy)(nil)
	if known != nil {
		p, err := n.VerifyPolicy(known)
		if err != nil {
			return membership.Proof{}, nil, err
		}
		pol = p
	} else {
		var err error
		if signed, pol, err = fetchPolicy(ctx, n, bootstrap); err != nil {
			return membership.Proof{}, nil, err
		}
	}
	if !pol.Open {
		return membership.Proof{}, nil, errors.New("the network is closed to nodes without a certificate")
	}
	pr, err := membership.LoadProof(path)
	switch {
	case err == nil && n.Work(id, pr) >= pol.PowBits:
		return pr, signed, nil
	case err != nil && !errors.Is(err, fs.ErrNotExist):
		return pr, nil, err
	}
	fmt.Printf("no certificate: joining as a guest; proving %d bits of work...\n", pol.PowBits)
	start := time.Now()
	if pr, err = n.Solve(ctx, id, pol.PowBits); err != nil {
		return pr, nil, err
	}
	fmt.Printf("proof of work found in %s\n", time.Since(start).Round(time.Millisecond))
	return pr, signed, membership.SaveProof(path, pr)
}

// fetchPolicy asks bootstrap nodes for the policy from a throwaway host: a
// connection that does not authenticate gets its peer briefly refused.
func fetchPolicy(ctx context.Context, n *membership.Network, bootstrap []peer.AddrInfo) (*membership.SignedPolicy, *membership.Policy, error) {
	if len(bootstrap) == 0 {
		return nil, nil, errors.New("no certificate, no -policy file, and no -bootstrap node to ask for the network policy")
	}
	h, err := libp2p.New(libp2p.NoListenAddrs)
	if err != nil {
		return nil, nil, err
	}
	defer h.Close()
	var lastErr error
	for _, ai := range bootstrap {
		ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		s, p, err := mesh.FetchPolicy(ctx, h, n, ai)
		cancel()
		if err == nil {
			return s, p, nil
		}
		lastErr = fmt.Errorf("network policy from %s: %w", ai.ID.ShortString(), err)
	}
	return nil, nil, lastErr
}
