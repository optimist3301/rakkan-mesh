package mesh

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

// policyProtocol serves the network policy to anyone, before
// authentication: it is signed by the root, so a node about to join as a
// guest can learn the work it must prove from any node, trusted or not.
const policyProtocol = protocol.ID("/rakkan/policy/1.0.0")

// policyDropGrace is how long a guest the new policy no longer admits stays
// connected, so that the policy can reach it.
const policyDropGrace = time.Second

func (n *Node) policyTopic() string { return "/rakkan/" + n.tag() + "/policy/1" }

func (n *Node) startPolicyGossip() error {
	topic := n.policyTopic()
	err := n.ps.RegisterTopicValidator(topic, func(_ context.Context, _ peer.ID, m *pubsub.Message) pubsub.ValidationResult {
		s, err := membership.DecodePolicy(m.Data)
		if err != nil {
			return pubsub.ValidationReject
		}
		p, err := n.cfg.Network.VerifyPolicy(s)
		if err != nil {
			return pubsub.ValidationReject
		}
		if _, cur := n.policies.Current(); p.Seq < cur.Seq {
			return pubsub.ValidationIgnore
		}
		return pubsub.ValidationAccept
	})
	if err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	if n.policy, err = n.ps.Join(topic); err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	sub, err := n.policy.Subscribe()
	if err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	n.wg.Go(func() {
		for {
			m, err := sub.Next(n.ctx)
			if err != nil {
				return
			}
			if s, err := membership.DecodePolicy(m.Data); err == nil {
				n.applyPolicy(s)
			}
		}
	})
	return nil
}

// Policy returns the network policy in force; without one the network is
// closed.
func (n *Node) Policy() membership.Policy {
	_, p := n.policies.Current()
	return p
}

// PublishPolicy applies a policy locally and spreads it to the network.
func (n *Node) PublishPolicy(ctx context.Context, s *membership.SignedPolicy) error {
	changed, err := n.applyPolicy(s)
	if err != nil {
		return err
	}
	if !changed {
		return errors.New("policy is not newer than the current one")
	}
	return n.policy.Publish(ctx, s.Encode())
}

// applyPolicy adopts a newer policy and drops guests it no longer admits.
// A guest whose own proof falls short solves a new one and reconnects.
func (n *Node) applyPolicy(s *membership.SignedPolicy) (bool, error) {
	changed, err := n.policies.Update(s)
	if err != nil || !changed {
		return changed, err
	}
	pol := n.Policy()
	n.log.Info("network policy updated", "seq", pol.Seq, "open", pol.Open, "pow_bits", pol.PowBits)
	n.mu.Lock()
	var dropped []peer.ID
	for p, st := range n.peers {
		if st.authed && st.proof != nil && n.recheck(st, p) != nil {
			dropped = append(dropped, p)
		}
	}
	n.mu.Unlock()
	// Give the policy a moment to reach the guests on its way out: they
	// learn the new difficulty from it, and a dropped connection would lose
	// the message still queued for them.
	for _, p := range dropped {
		n.wg.Go(func() {
			select {
			case <-time.After(policyDropGrace):
				n.deny(p, denyAfterFailure, errors.New("the policy no longer admits this guest"))
			case <-n.ctx.Done():
			}
		})
	}
	if n.cfg.Cert == nil && pol.Open {
		n.mu.Lock()
		weak := n.cfg.Network.Work(n.host.ID(), n.proof) < pol.PowBits
		n.mu.Unlock()
		if weak {
			n.wg.Go(func() { n.resolve(pol.PowBits) })
		}
	}
	return true, nil
}

// resolve finds a proof for a raised difficulty and reconnects with it.
func (n *Node) resolve(bits int) {
	n.log.Warn("the network wants more proof of work; solving", "bits", bits)
	pr, err := n.cfg.Network.Solve(n.ctx, n.host.ID(), bits)
	if err != nil {
		return
	}
	n.mu.Lock()
	if n.cfg.Network.Work(n.host.ID(), n.proof) < bits {
		n.proof = pr
	}
	n.mu.Unlock()
	if n.cfg.OnProof != nil {
		n.cfg.OnProof(pr)
	}
	for _, m := range n.Members() {
		if m.Connected {
			go n.authenticate(m.ID)
		}
	}
}

func (n *Node) handlePolicy(s network.Stream) {
	defer s.Close()
	s.SetDeadline(time.Now().Add(10 * time.Second))
	if cur, _ := n.policies.Current(); cur != nil {
		s.Write(cur.Encode())
	}
}

// FetchPolicy asks a node for the network policy and checks the root's
// signature. A node joining as a guest uses it to learn the work to prove;
// it dials from h, a throwaway host, since a connection that does not
// authenticate gets its peer briefly refused.
func FetchPolicy(ctx context.Context, h interface {
	Connect(context.Context, peer.AddrInfo) error
	NewStream(context.Context, peer.ID, ...protocol.ID) (network.Stream, error)
}, nw *membership.Network, ai peer.AddrInfo) (*membership.SignedPolicy, *membership.Policy, error) {
	if err := h.Connect(ctx, ai); err != nil {
		return nil, nil, err
	}
	s, err := h.NewStream(ctx, ai.ID, policyProtocol)
	if err != nil {
		return nil, nil, err
	}
	defer s.Close()
	if d, ok := ctx.Deadline(); ok {
		s.SetDeadline(d)
	}
	b, err := readAllLimited(s, maxHello)
	if err != nil {
		return nil, nil, err
	}
	if len(b) == 0 {
		return nil, nil, errClosed
	}
	sp, err := membership.DecodePolicy(b)
	if err != nil {
		return nil, nil, err
	}
	p, err := nw.VerifyPolicy(sp)
	if err != nil {
		return nil, nil, err
	}
	return sp, p, nil
}

func readAllLimited(s network.Stream, limit int) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(s, int64(limit)+1))
	if len(b) > limit {
		return nil, fmt.Errorf("reply over %d bytes", limit)
	}
	return b, err
}
