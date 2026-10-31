package mesh

import (
	"context"
	"errors"
	"fmt"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

func (n *Node) crlTopic() string { return "/rakkan/" + n.tag() + "/crl/1" }

func (n *Node) startCRLGossip() error {
	topic := n.crlTopic()
	err := n.ps.RegisterTopicValidator(topic, func(_ context.Context, _ peer.ID, m *pubsub.Message) pubsub.ValidationResult {
		s, err := membership.DecodeCRL(m.Data)
		if err != nil {
			return pubsub.ValidationReject
		}
		c, err := n.cfg.Network.VerifyCRL(s)
		if err != nil {
			return pubsub.ValidationReject
		}
		// Older lists are stale, not malicious. The current one passes:
		// this node validates its own publications after applying them,
		// and pubsub drops duplicate deliveries anyway.
		if _, seq := n.revs.Current(); c.Seq < seq {
			return pubsub.ValidationIgnore
		}
		return pubsub.ValidationAccept
	})
	if err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	if n.crl, err = n.ps.Join(topic); err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	sub, err := n.crl.Subscribe()
	if err != nil {
		return fmt.Errorf("mesh: %w", err)
	}
	n.wg.Go(func() {
		for {
			m, err := sub.Next(n.ctx)
			if err != nil {
				return
			}
			if s, err := membership.DecodeCRL(m.Data); err == nil {
				n.applyCRL(s)
			}
		}
	})
	return nil
}

// CRLSeq returns the sequence number of the newest revocation list known.
func (n *Node) CRLSeq() uint64 {
	_, seq := n.revs.Current()
	return seq
}

// PublishCRL applies a revocation list locally and spreads it to members.
func (n *Node) PublishCRL(ctx context.Context, s *membership.SignedCRL) error {
	changed, err := n.applyCRL(s)
	if err != nil {
		return err
	}
	if !changed {
		return errors.New("revocation list is not newer than the current one")
	}
	return n.crl.Publish(ctx, s.Encode())
}

// applyCRL adopts a newer revocation list and drops members it revokes.
func (n *Node) applyCRL(s *membership.SignedCRL) (bool, error) {
	changed, err := n.revs.Update(s)
	if err != nil || !changed {
		return changed, err
	}
	_, seq := n.revs.Current()
	n.log.Info("revocation list updated", "seq", seq)
	if n.cfg.Cert != nil {
		if _, err := n.cfg.Network.Verify(n.cfg.Cert, time.Now(), n.revs.Revoked); err != nil {
			n.log.Error("this node's certificate is revoked", "err", err)
		}
	} else if n.revs.Revoked(n.host.ID().String()) {
		n.log.Error("this node is revoked")
	}
	n.mu.Lock()
	var revoked []peer.ID
	for p, st := range n.peers {
		if err := n.recheck(st, p); errors.Is(err, membership.ErrRevoked) {
			revoked = append(revoked, p)
			st.dial = false
		}
	}
	n.mu.Unlock()
	for _, p := range revoked {
		n.deny(p, 100*365*24*time.Hour, membership.ErrRevoked)
	}
	return true, nil
}
