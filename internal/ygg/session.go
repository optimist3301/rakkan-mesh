package ygg

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"slices"
	"sync"
	"time"

	iwt "github.com/Arceliar/ironwood/types"
)

// Before QUIC starts, a dial makes sure the overlay can carry it both ways:
// first a path to the other node (waitPath), then a session that answers a
// ping (waitSession). QUIC sent earlier would be lost, and its
// retransmissions back off exponentially: between two new nodes the first
// connection took up to half a minute without this.
//
// Pings and pongs are one-byte datagrams whose two top bits are zero:
// quic-go hands those over as non-QUIC packets (ctlLoop) instead of
// parsing them. Its transport reads only once started; reading them
// starts it, so a node that has not listened or dialled yet answers too.
const (
	ctlPing = 1
	ctlPong = 2
)

// lookupEvery is how often a dial repeats its path lookup and its ping.
const lookupEvery = time.Second

// ctlLoop answers pings and hands pongs to waitSession.
func (n *Node) ctlLoop() {
	defer n.wg.Done()
	buf := make([]byte, 64)
	for {
		k, from, err := n.tr.ReadNonQUICPacket(n.ctx, buf)
		if err != nil {
			return
		}
		key, ok := from.(iwt.Addr)
		if !ok || k != 1 {
			continue
		}
		switch buf[0] {
		case ctlPing:
			n.tr.WriteTo([]byte{ctlPong}, from)
		case ctlPong:
			n.pongs.notify(ed25519.PublicKey(key))
		}
	}
}

// waitPath asks the overlay for a path to key and waits until one is known.
// A path found earlier raises no notification again, so the known paths
// are checked too.
func (n *Node) waitPath(ctx context.Context, key ed25519.PublicKey) error {
	ch, done := n.paths.wait(key)
	defer done()
	t := time.NewTicker(lookupEvery)
	defer t.Stop()
	for {
		if n.hasPath(key) {
			return nil
		}
		n.core.SendLookup(key)
		select {
		case <-ch:
			return nil
		case <-ctx.Done():
			return fmt.Errorf("no path to %x over Yggdrasil yet: %w", key[:8], ctx.Err())
		case <-t.C:
		}
	}
}

// waitSession pings key until it answers: the session between the two
// nodes then works both ways.
func (n *Node) waitSession(ctx context.Context, key ed25519.PublicKey) error {
	ch, done := n.pongs.wait(key)
	defer done()
	t := time.NewTicker(lookupEvery / 2)
	defer t.Stop()
	for {
		n.tr.WriteTo([]byte{ctlPing}, iwt.Addr(key))
		select {
		case <-ch:
			return nil
		case <-ctx.Done():
			return fmt.Errorf("no answer from %x over Yggdrasil yet: %w", key[:8], ctx.Err())
		case <-t.C:
		}
	}
}

func (n *Node) hasPath(key ed25519.PublicKey) bool {
	for _, p := range n.core.GetPaths() {
		if key.Equal(p.Key) {
			return true
		}
	}
	return false
}

// nudgeFor is how long a new session is nudged, every nudgeEvery.
const (
	nudgeEvery = 500 * time.Millisecond
	nudgeFor   = 15 * time.Second
)

// nudgeLoop pings over every session in its first nudgeFor. A session the
// other side opened stalls half-open if this side's first answer to it is
// lost: an answer goes out once, and the other side's repeated requests
// carry the same sequence number, which this side ignores. Only traffic
// from this side mends it, and a QUIC server sends none until it has heard
// from the client. The keys of such sessions are known only from
// GetSessions.
//
// GetSessions reads each session's byte counters without synchronisation
// (ironwood's encrypted/debug.go), a race on two counters this package does
// not use. The race detector reports it, so builds with -race leave the
// loop out (race_on.go).
func (n *Node) nudgeLoop() {
	defer n.wg.Done()
	if !nudgeSessions {
		return
	}
	t := time.NewTicker(nudgeEvery)
	defer t.Stop()
	for {
		select {
		case <-n.ctx.Done():
			return
		case <-t.C:
		}
		for _, s := range n.core.GetSessions() {
			if s.Uptime < nudgeFor {
				n.tr.WriteTo([]byte{ctlPing}, iwt.Addr(s.Key))
			}
		}
	}
}

// signals wakes goroutines waiting for an event about a key.
type signals struct {
	mu      sync.Mutex
	waiting map[[ed25519.PublicKeySize]byte][]chan struct{}
}

// wait returns a channel that receives when notify is called for key, and
// a function to call once done waiting.
func (s *signals) wait(key ed25519.PublicKey) (<-chan struct{}, func()) {
	k := [ed25519.PublicKeySize]byte(key)
	ch := make(chan struct{}, 1)
	s.mu.Lock()
	if s.waiting == nil {
		s.waiting = map[[ed25519.PublicKeySize]byte][]chan struct{}{}
	}
	s.waiting[k] = append(s.waiting[k], ch)
	s.mu.Unlock()
	return ch, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		w := s.waiting[k]
		if i := slices.Index(w, ch); i >= 0 {
			w = slices.Delete(w, i, i+1)
		}
		if len(w) == 0 {
			delete(s.waiting, k)
		} else {
			s.waiting[k] = w
		}
	}
}

func (s *signals) notify(key ed25519.PublicKey) {
	if len(key) != ed25519.PublicKeySize {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, ch := range s.waiting[[ed25519.PublicKeySize]byte(key)] {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}
