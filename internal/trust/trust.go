// Package trust decides how far a node relies on its peers in an open
// network, where guests join without a certificate (package membership).
//
// Members are trusted: the root vouched for them. A guest earns trust by
// returning results that independent checks confirm, and loses it, along
// with its connection, when caught returning a wrong one. Every node keeps
// its own evidence: results it verified, failures, cheating, and the gas
// each peer spent computing for it and it for them.
//
// Reputation is EigenTrust (Kamvar et al., 2003). Each node gossips its
// local trust: how its verified dealings split among peers. Every node then
// computes the global trust vector t = (1-a)·Cᵀt + a·p, where row i of C is
// node i's normalized local trust and p, the pre-trusted distribution, is
// spread over members. Trust flows only from members and from peers they
// trust, weighted by the trust in whoever reports it, so a clique of fresh
// identities praising each other gains nothing.
//
// Credit is bilateral, like BitTorrent's tit-for-tat: a guest may have a
// node compute for it up to a free allowance more than it computed for that
// node, plus a credit line proportional to its global trust. Contributing
// to the network thus buys service from all of it, through reputation,
// without a shared ledger. Members are not limited.
package trust

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"slices"
	"sync"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/mesh"
)

// ErrNoCredit refuses work for a guest that has used up its credit here.
var ErrNoCredit = errors.New("not enough credit")

// Evidence weights: a lie outweighs many honest answers, and a failure,
// which may be bad luck, costs as much as one honest answer earns.
const (
	weightVerified = 1
	weightFailed   = 1
	weightCheated  = 20
)

// alpha is EigenTrust's teleport weight toward the pre-trusted peers.
const alpha = 0.15

// maxReported bounds the peers in a gossiped local trust vector.
const maxReported = 128

// Config configures a Service.
type Config struct {
	Node *mesh.Node
	// Path persists the ledger across restarts; empty keeps it in memory.
	Path string
	// FreeCredit is the gas a guest may have this node spend for it
	// beyond what it spent for this node. Default 50e9: ten tasks at the
	// default gas limit, about eight seconds of CPU.
	FreeCredit int64
	// CreditPerTrust is the further credit of a guest with full trust.
	// Default 100 times FreeCredit.
	CreditPerTrust int64
	// MinVerify is the share of results from fully trusted guests still
	// checked. Default 0.1; a new guest's results are all checked.
	MinVerify float64
	// BanFor is how long a guest caught cheating is refused. Default 24h.
	BanFor time.Duration
	// GossipInterval is how often local trust is published. Default 30s.
	GossipInterval time.Duration
	Logger         *slog.Logger
}

// Peer is what this node knows of a peer.
type Peer struct {
	ID peer.ID `json:"id"`
	// Served is the gas this node spent for the peer, Consumed what the
	// peer spent for this node.
	Served   int64 `json:"served"`
	Consumed int64 `json:"consumed"`
	Verified int64 `json:"verified"`
	Failed   int64 `json:"failed"`
	Cheated  int64 `json:"cheated"`
	// LastCheated is when the peer was last caught.
	LastCheated time.Time `json:"last_cheated,omitzero"`

	reserved int64 // gas of the peer's tasks running here
}

// satisfaction is the peer's standing in this node's eyes.
func (p *Peer) satisfaction() float64 {
	return max(float64(p.Verified*weightVerified-p.Failed*weightFailed-p.Cheated*weightCheated), 0)
}

// Service keeps the ledger and computes reputation.
type Service struct {
	cfg   Config
	node  *mesh.Node
	log   *slog.Logger
	topic *pubsub.Topic

	mu       sync.Mutex
	peers    map[peer.ID]*Peer
	reports  map[peer.ID]report // local trust gossiped by others
	global   map[peer.ID]float64
	computed time.Time
	dirty    bool

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

type report struct {
	trust    map[peer.ID]float64
	received time.Time
}

// wireReport is a gossiped local trust vector.
type wireReport struct {
	Version int                `json:"v"`
	Trust   map[string]float64 `json:"trust"`
}

// New starts a service on a running node.
func New(cfg Config) (*Service, error) {
	if cfg.Node == nil {
		return nil, errors.New("trust: node is required")
	}
	if cfg.FreeCredit <= 0 {
		cfg.FreeCredit = 50e9
	}
	if cfg.CreditPerTrust <= 0 {
		cfg.CreditPerTrust = 100 * cfg.FreeCredit
	}
	if cfg.MinVerify <= 0 {
		cfg.MinVerify = 0.1
	}
	if cfg.BanFor <= 0 {
		cfg.BanFor = 24 * time.Hour
	}
	if cfg.GossipInterval <= 0 {
		cfg.GossipInterval = 30 * time.Second
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	s := &Service{cfg: cfg, node: cfg.Node, log: cfg.Logger, peers: map[peer.ID]*Peer{},
		reports: map[peer.ID]report{}, global: map[peer.ID]float64{}, dirty: true}
	if err := s.load(); err != nil {
		return nil, err
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	var err error
	if s.topic, err = cfg.Node.JoinTopic("trust/1"); err != nil {
		return nil, fmt.Errorf("trust: %w", err)
	}
	sub, err := s.topic.Subscribe()
	if err != nil {
		return nil, fmt.Errorf("trust: %w", err)
	}
	s.wg.Go(func() { s.receive(sub) })
	s.wg.Go(s.publishLoop)
	return s, nil
}

// Close saves the ledger and stops gossip.
func (s *Service) Close() error {
	s.cancel()
	s.wg.Wait()
	return s.save()
}

func (s *Service) peer(p peer.ID) *Peer {
	st, ok := s.peers[p]
	if !ok {
		st = &Peer{ID: p}
		s.peers[p] = st
	}
	return st
}

// Reserve admits a task from p that may spend up to gas here, or refuses
// it with ErrNoCredit. The returned function settles it with the gas the
// task actually used.
func (s *Service) Reserve(p peer.ID, gas int64) (settle func(used int64), err error) {
	certified := s.node.IsCertified(p)
	score := s.Score(p)
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.peer(p)
	if !certified {
		limit := s.cfg.FreeCredit + int64(score*float64(s.cfg.CreditPerTrust))
		if debt := st.Served - st.Consumed + st.reserved; debt+gas > limit {
			return nil, fmt.Errorf("%w: owes %d gas, asks for %d more, credit %d", ErrNoCredit, max(debt, 0), gas, limit)
		}
	}
	st.reserved += gas
	var once sync.Once
	return func(used int64) {
		once.Do(func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			st.reserved -= gas
			st.Served += min(max(used, 0), gas)
		})
	}, nil
}

// Consumed records gas p spent computing for this node.
func (s *Service) Consumed(p peer.ID, gas int64) {
	if gas <= 0 || p == s.node.ID() {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.peer(p).Consumed += gas
}

// Verified records a result from p that an independent one confirmed.
func (s *Service) Verified(p peer.ID) {
	if p == s.node.ID() {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.peer(p).Verified++
	s.dirty = true
}

// Failed records that p did not deliver a result it took on.
func (s *Service) Failed(p peer.ID) {
	if p == s.node.ID() {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.peer(p).Failed++
	s.dirty = true
}

// Cheated records a wrong result from p, caught by comparison with results
// from others. A guest is disconnected and refused for BanFor; a member,
// whom only the root can expel, is no longer given work for that long.
func (s *Service) Cheated(p peer.ID, what string) {
	if p == s.node.ID() {
		return
	}
	s.mu.Lock()
	st := s.peer(p)
	st.Cheated++
	st.LastCheated = time.Now()
	s.dirty = true
	s.mu.Unlock()
	if s.node.IsCertified(p) {
		s.log.Error("member returned a wrong result; consider revoking it", "peer", p, "what", what)
		return
	}
	s.log.Warn("guest returned a wrong result; banning it", "peer", p, "what", what, "for", s.cfg.BanFor)
	s.node.Ban(p, s.cfg.BanFor, fmt.Errorf("wrong result: %s", what))
}

// Suspect reports whether p was caught cheating within BanFor; such peers
// get no work.
func (s *Service) Suspect(p peer.ID) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.peers[p]
	return ok && !st.LastCheated.IsZero() && time.Since(st.LastCheated) < s.cfg.BanFor
}

// Trusted reports whether p's results are taken without checking: p is
// this node or a member.
func (s *Service) Trusted(p peer.ID) bool {
	return p == s.node.ID() || s.node.IsCertified(p)
}

// ShouldVerify draws whether to check a result from p: always for a new
// guest, down to MinVerify for a fully trusted one, never for members.
func (s *Service) ShouldVerify(p peer.ID) bool {
	if s.Trusted(p) {
		return false
	}
	return rand.Float64() < max(s.cfg.MinVerify, 1-s.Score(p))
}

// Score is p's reputation from 0 to 1: 1 for this node and members; for a
// guest, its global trust relative to the average share, capped at 1.
// Members are not the yardstick: nobody verifies them, so trust flows to
// them only by teleport. A sybil clique holds about what leaked to it in
// all, however many identities split it, so its members score near zero
// while honest guests, as trusted as the average peer, score 1.
func (s *Service) Score(p peer.ID) float64 {
	if s.Trusted(p) {
		return 1
	}
	g := s.globalTrust()
	n := 0
	for _, t := range g {
		if t > 1e-12 {
			n++
		}
	}
	return min(g[p]*float64(n), 1)
}

// Peers lists the ledger, with each peer's score.
func (s *Service) Peers() []PeerInfo {
	s.mu.Lock()
	var out []PeerInfo
	for _, st := range s.peers {
		out = append(out, PeerInfo{Peer: *st})
	}
	s.mu.Unlock()
	for i := range out {
		out[i].Score = s.Score(out[i].ID)
		out[i].Member = s.Trusted(out[i].ID)
	}
	slices.SortFunc(out, func(a, b PeerInfo) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		return cmp.Compare(a.ID, b.ID)
	})
	return out
}

// PeerInfo is a peer's ledger entry and reputation.
type PeerInfo struct {
	Peer
	Score  float64 `json:"score"`
	Member bool    `json:"member"`
}
