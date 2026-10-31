package trust

import (
	"cmp"
	"encoding/json"
	"errors"
	"io/fs"
	"maps"
	"os"
	"slices"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/peer"
)

// reportTTL drops local trust vectors not refreshed by their author.
const reportTTL = 10 * time.Minute

// saturation is the satisfaction at which a node's local trust is fully
// given out. Below it, the rest stays with the pre-trusted members, so a
// handful of honest answers earns a little trust rather than all this node
// has to give: reputation reflects the volume of verified work, not only
// its share.
const saturation = 100

// localTrust is this node's local trust: each peer's share of its
// satisfaction, or of saturation while the total is below it, for the
// heaviest maxReported peers.
func (s *Service) localTrust() map[peer.ID]float64 {
	type kv struct {
		p peer.ID
		v float64
	}
	var all []kv
	for p, st := range s.peers {
		if v := st.satisfaction(); v > 0 {
			all = append(all, kv{p, v})
		}
	}
	slices.SortFunc(all, func(a, b kv) int {
		if c := cmp.Compare(b.v, a.v); c != 0 {
			return c
		}
		return cmp.Compare(a.p, b.p)
	})
	if len(all) > maxReported {
		all = all[:maxReported]
	}
	var total float64
	for _, e := range all {
		total += e.v
	}
	sum := max(total, saturation)
	out := make(map[peer.ID]float64, len(all))
	for _, e := range all {
		out[e.p] = e.v / sum
	}
	return out
}

// recompute bounds how stale the global trust vector gets while members
// come and go.
const recompute = 2 * time.Second

// globalTrust returns the global trust vector, recomputing it when evidence
// or reports changed.
func (s *Service) globalTrust() map[peer.ID]float64 {
	pretrusted := []peer.ID{}
	if s.node.Certified() {
		pretrusted = append(pretrusted, s.node.ID())
	}
	for _, m := range s.node.Members() {
		if !m.Guest {
			pretrusted = append(pretrusted, m.ID)
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.dirty && time.Since(s.computed) < recompute {
		return s.global
	}
	rows := map[peer.ID]map[peer.ID]float64{s.node.ID(): s.localTrust()}
	for p, r := range s.reports {
		if time.Since(r.received) > reportTTL {
			delete(s.reports, p)
			continue
		}
		rows[p] = r.trust
	}
	s.global = eigenTrust(rows, pretrusted)
	s.dirty, s.computed = false, time.Now()
	return s.global
}

// eigenTrust iterates t = (1-alpha)·Cᵀt + alpha·p to convergence. rows are
// the local trust vectors, each summing to at most 1; a peer without one,
// or with an empty one, defers to the pre-trusted distribution p. With no
// pre-trusted peers, nobody is trusted.
func eigenTrust(rows map[peer.ID]map[peer.ID]float64, pretrusted []peer.ID) map[peer.ID]float64 {
	if len(pretrusted) == 0 {
		return map[peer.ID]float64{}
	}
	pre := map[peer.ID]float64{}
	for _, p := range pretrusted {
		pre[p] = 1 / float64(len(pretrusted))
	}
	t := maps.Clone(pre)
	for range 50 {
		next := map[peer.ID]float64{}
		var lost float64 // trust held by peers with no opinion
		for i, ti := range t {
			row := rows[i]
			var given float64
			for j, c := range row {
				if j != i && c > 0 {
					next[j] += (1 - alpha) * ti * c
					given += c
				}
			}
			lost += (1 - alpha) * ti * max(1-given, 0)
		}
		for p, v := range pre {
			next[p] += (alpha + lost) * v
		}
		var delta float64
		for p, v := range next {
			d := v - t[p]
			if d < 0 {
				d = -d
			}
			delta += d
		}
		t = next
		if delta < 1e-9 {
			break
		}
	}
	return t
}

func (s *Service) publishLoop() {
	tick := time.NewTicker(s.cfg.GossipInterval)
	defer tick.Stop()
	for {
		s.publish()
		select {
		case <-s.ctx.Done():
			return
		case <-tick.C:
			if err := s.save(); err != nil {
				s.log.Warn("saving trust ledger", "err", err)
			}
		}
	}
}

func (s *Service) publish() {
	s.mu.Lock()
	local := s.localTrust()
	s.mu.Unlock()
	if len(local) == 0 {
		return
	}
	w := wireReport{Version: 1, Trust: map[string]float64{}}
	for p, v := range local {
		w.Trust[p.String()] = v
	}
	b, _ := json.Marshal(w)
	s.topic.Publish(s.ctx, b)
}

func (s *Service) receive(sub *pubsub.Subscription) {
	for {
		m, err := sub.Next(s.ctx)
		if err != nil {
			return
		}
		from := m.GetFrom()
		if from == s.node.ID() || !s.node.IsMember(from) || len(m.Data) > 64<<10 {
			continue
		}
		var w wireReport
		if json.Unmarshal(m.Data, &w) != nil || w.Version != 1 || len(w.Trust) > maxReported {
			continue
		}
		r := report{trust: map[peer.ID]float64{}, received: time.Now()}
		var sum float64
		for k, v := range w.Trust {
			p, err := peer.Decode(k)
			if err != nil || !(v > 0) || v > 1 {
				continue
			}
			r.trust[p] = v
			sum += v
		}
		if sum > 1 {
			for p := range r.trust {
				r.trust[p] /= sum
			}
		}
		s.mu.Lock()
		s.reports[from] = r
		s.dirty = true
		s.mu.Unlock()
	}
}

// report records a peer's gossiped local trust, for tests.
func (s *Service) report(from peer.ID, trust map[peer.ID]float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reports[from] = report{trust: trust, received: time.Now()}
	s.dirty = true
}

func (s *Service) load() error {
	if s.cfg.Path == "" {
		return nil
	}
	b, err := os.ReadFile(s.cfg.Path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var peers []Peer
	if err := json.Unmarshal(b, &peers); err != nil {
		return err
	}
	for _, p := range peers {
		s.peers[p.ID] = &p
	}
	return nil
}

func (s *Service) save() error {
	if s.cfg.Path == "" {
		return nil
	}
	s.mu.Lock()
	peers := make([]Peer, 0, len(s.peers))
	for _, p := range s.peers {
		peers = append(peers, *p)
	}
	s.mu.Unlock()
	b, err := json.MarshalIndent(peers, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.cfg.Path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.cfg.Path)
}
