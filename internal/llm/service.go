package llm

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/trust"
)

// Protocol serves pipeline stages. One stream is one stage of one
// generation, and its key/value cache lives exactly as long as the stream:
//
//	-> {"op":"open","model":id,"from":l0,"to":l1,"ctx":n,"providers":[...]}  <- status
//	-> {"op":"forward","pos":p,"tokens":t} + activations [t][dim]    <- status + activations
//
// The node fetches the layers' blobs from the providers if needed. A
// generation's cache is reachable only through its stream, so members
// cannot read each other's.
const Protocol = protocol.ID("/rakkan/llm/1.0.0")

// Capability is advertised by nodes serving stages.
const Capability = "llm/1"

const maxRequest = 64 << 10

type request struct {
	Op        string   `json:"op"`
	Model     string   `json:"model,omitempty"`
	From      int      `json:"from,omitempty"`
	To        int      `json:"to,omitempty"`
	Providers []string `json:"providers,omitempty"`
	Pos       int      `json:"pos,omitempty"`
	Tokens    int      `json:"tokens,omitempty"`
	// Ctx is the positions a generation needs, for which the stage keeps
	// its cache; 0 means the model's whole context.
	Ctx int `json:"ctx,omitempty"`
}

type status struct {
	OK    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

// Shard is a model's layer range loaded on a node, as advertised.
type Shard struct {
	Model string `json:"model"`
	From  int    `json:"from"`
	To    int    `json:"to"`
}

type advert struct {
	Serves   bool    `json:"serves"`
	Shards   []Shard `json:"shards"`
	FreeMem  int64   `json:"free_mem"`
	received time.Time
}

// ServiceConfig configures a Service.
type ServiceConfig struct {
	Node  *mesh.Node
	Blobs *blob.Service
	// Serve enables serving stages for other members; MemoryBytes bounds
	// the weights and caches held for them. Default 4 GiB.
	Serve       bool
	MemoryBytes int64
	// Trust, when set, charges guests' credit for the stages this node
	// serves them, and has stages on guests checked by a witness (see
	// Generate). Without it every peer is trusted.
	Trust  *trust.Service
	Logger *slog.Logger
}

// Service serves pipeline stages and coordinates generations.
type Service struct {
	cfg   ServiceConfig
	node  *mesh.Node
	blobs *blob.Service
	log   *slog.Logger
	topic *pubsub.Topic

	mu       sync.Mutex
	shards   map[Shard]*loaded
	used     int64 // bytes of weights and caches held
	adverts  map[peer.ID]advert
	coords   map[blob.ID]*coordModel
	ctx      context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	changed  chan struct{}
	failNext func(Shard) bool       // for tests: fail a stage's next forward
	tamper   func(Shard, []float32) // for tests: alter a stage's outputs
}

type loaded struct {
	manifest *Manifest
	layers   []*Layer
	rope     *Rope
	bytes    int64
	users    int
	lastUsed time.Time
	ready    chan struct{} // closed once loaded
	err      error
}

// NewService starts a service.
func NewService(cfg ServiceConfig) (*Service, error) {
	if cfg.MemoryBytes <= 0 {
		cfg.MemoryBytes = 4 << 30
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	s := &Service{
		cfg: cfg, node: cfg.Node, blobs: cfg.Blobs, log: cfg.Logger,
		shards: map[Shard]*loaded{}, adverts: map[peer.ID]advert{}, coords: map[blob.ID]*coordModel{},
		changed: make(chan struct{}, 1),
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	var err error
	if s.topic, err = cfg.Node.JoinTopic("llm/1"); err != nil {
		return nil, err
	}
	sub, err := s.topic.Subscribe()
	if err != nil {
		return nil, err
	}
	s.wg.Go(func() { s.receiveAdverts(sub) })
	s.wg.Go(s.publishAdverts)
	if cfg.Serve {
		cfg.Node.Handle(Protocol, s.handle)
	}
	return s, nil
}

func (s *Service) Close() {
	s.cancel()
	s.wg.Wait()
}

func (s *Service) advert() advert {
	s.mu.Lock()
	defer s.mu.Unlock()
	a := advert{Serves: s.cfg.Serve, FreeMem: s.cfg.MemoryBytes - s.used}
	for sh, l := range s.shards {
		if l.err == nil && l.layers != nil {
			a.Shards = append(a.Shards, sh)
		}
	}
	return a
}

func (s *Service) notify() {
	select {
	case s.changed <- struct{}{}:
	default:
	}
}

func (s *Service) publishAdverts() {
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-tick.C:
		case <-s.changed:
		}
		b, _ := json.Marshal(s.advert())
		s.topic.Publish(s.ctx, b)
	}
}

func (s *Service) receiveAdverts(sub *pubsub.Subscription) {
	for {
		m, err := sub.Next(s.ctx)
		if err != nil {
			return
		}
		from := m.GetFrom()
		if from == s.node.ID() || !s.node.IsMember(from) {
			continue
		}
		var a advert
		if json.Unmarshal(m.Data, &a) != nil {
			continue
		}
		a.received = time.Now()
		s.mu.Lock()
		s.adverts[from] = a
		s.mu.Unlock()
	}
}

// servers returns members serving stages, by recent adverts.
func (s *Service) servers() map[peer.ID]advert {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[peer.ID]advert{}
	for p, a := range s.adverts {
		if a.Serves && time.Since(a.received) < 10*time.Second && s.node.IsMember(p) {
			out[p] = a
		}
	}
	return out
}

// fetchManifest makes a model's manifest available and parses it.
func (s *Service) fetchManifest(ctx context.Context, model blob.ID, providers []peer.ID) (*Manifest, error) {
	if err := s.blobs.Fetch(ctx, model, providers, maxManifest); err != nil {
		return nil, err
	}
	data, err := s.blobs.Store().ReadAll(model, maxManifest)
	if err != nil {
		return nil, err
	}
	return ParseManifest(data)
}

// acquire loads a shard (or waits for a concurrent load) and holds it.
func (s *Service) acquire(ctx context.Context, model blob.ID, from, to int, providers []peer.ID) (*loaded, error) {
	key := Shard{model.String(), from, to}
	s.mu.Lock()
	l, ok := s.shards[key]
	if !ok {
		l = &loaded{ready: make(chan struct{})}
		s.shards[key] = l
		s.mu.Unlock()
		l.manifest, l.layers, l.rope, l.bytes, l.err = s.load(ctx, model, from, to, providers)
		s.mu.Lock()
		if l.err != nil {
			delete(s.shards, key)
		} else {
			s.used += l.bytes
		}
		close(l.ready)
		s.mu.Unlock()
		s.notify()
	} else {
		s.mu.Unlock()
	}
	select {
	case <-l.ready:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if l.err != nil {
		return nil, l.err
	}
	s.mu.Lock()
	l.users++
	l.lastUsed = time.Now()
	s.mu.Unlock()
	return l, nil
}

func (s *Service) release(l *loaded) {
	s.mu.Lock()
	l.users--
	l.lastUsed = time.Now()
	s.mu.Unlock()
}

func (s *Service) load(ctx context.Context, model blob.ID, from, to int, providers []peer.ID) (*Manifest, []*Layer, *Rope, int64, error) {
	m, err := s.fetchManifest(ctx, model, providers)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if from < 0 || to > m.Config.Layers || from >= to {
		return nil, nil, nil, 0, fmt.Errorf("layers %d..%d outside the model's %d", from, to, m.Config.Layers)
	}
	size := LayerBytes(m.Config) * int64(to-from)
	if err := s.reserve(size); err != nil {
		return nil, nil, nil, 0, err
	}
	var layers []*Layer
	for i := from; i < to; i++ {
		id, _ := blob.ParseID(m.Layers[i])
		if err := s.blobs.Fetch(ctx, id, providers, size); err != nil {
			s.unreserve(size)
			return nil, nil, nil, 0, err
		}
		data, err := s.blobs.Store().ReadAll(id, LayerBytes(m.Config))
		if err == nil {
			var l *Layer
			if l, err = LoadLayer(m.Config, data); err == nil {
				layers = append(layers, l)
			}
		}
		if err != nil {
			s.unreserve(size)
			return nil, nil, nil, 0, err
		}
	}
	s.unreserve(size) // accounted by the caller once loaded
	s.log.Info("loaded model layers", "model", model.String()[:12], "from", from, "to", to)
	return m, layers, NewRope(m.Config, m.Config.SeqLen), size, nil
}

// reserve checks the memory budget, unloading idle shards to make room.
func (s *Service) reserve(n int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.used+n > s.cfg.MemoryBytes {
		var victim Shard
		var oldest time.Time
		for k, l := range s.shards {
			if l.users == 0 && l.layers != nil && (oldest.IsZero() || l.lastUsed.Before(oldest)) {
				victim, oldest = k, l.lastUsed
			}
		}
		if oldest.IsZero() {
			return fmt.Errorf("model memory budget exhausted (%d of %d bytes in use)", s.used, s.cfg.MemoryBytes)
		}
		s.used -= s.shards[victim].bytes
		delete(s.shards, victim)
	}
	s.used += n
	return nil
}

func (s *Service) unreserve(n int64) {
	s.mu.Lock()
	s.used -= n
	s.mu.Unlock()
}

func writeFrame(w io.Writer, b []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func readFrame(r io.Reader, limit int) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if uint64(n) > uint64(limit) {
		return nil, fmt.Errorf("frame of %d bytes exceeds %d", n, limit)
	}
	b := make([]byte, n)
	_, err := io.ReadFull(r, b)
	return b, err
}

func writeStatus(w io.Writer, err error) error {
	st := status{OK: err == nil}
	if err != nil {
		st.Error = err.Error()
	}
	b, _ := json.Marshal(st)
	return writeFrame(w, b)
}

func readStatus(r io.Reader) error {
	b, err := readFrame(r, maxRequest)
	if err != nil {
		return err
	}
	var st status
	if err := json.Unmarshal(b, &st); err != nil {
		return err
	}
	if !st.OK {
		return errors.New(st.Error)
	}
	return nil
}

func encodeActs(xs []float32) []byte { return f32Bytes(xs) }

func decodeActs(b []byte) []float32 { return floats(b) }

// handle serves one stage of one generation.
func (s *Service) handle(st network.Stream, c *membership.Cert) {
	defer st.Close()
	st.SetReadDeadline(time.Now().Add(time.Minute))
	b, err := readFrame(st, maxRequest)
	if err != nil {
		st.Reset()
		return
	}
	var req request
	if json.Unmarshal(b, &req) != nil || req.Op != "open" {
		st.Reset()
		return
	}
	model, err := blob.ParseID(req.Model)
	if err != nil {
		writeStatus(st, err)
		return
	}
	providers := []peer.ID{c.Subject}
	for _, p := range req.Providers {
		if id, err := peer.Decode(p); err == nil && id != c.Subject {
			providers = append(providers, id)
		}
	}
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Minute)
	l, err := s.acquire(ctx, model, req.From, req.To, providers)
	cancel()
	if err != nil {
		writeStatus(st, err)
		return
	}
	defer s.release(l)
	cfg := l.manifest.Config
	positions := req.Ctx
	if positions <= 0 || positions > cfg.SeqLen {
		positions = cfg.SeqLen
	}
	cache := CacheBytes(cfg, req.To-req.From, positions)
	if err := s.reserve(cache); err != nil {
		writeStatus(st, err)
		return
	}
	defer func() { s.unreserve(cache); s.notify() }()
	stage := NewStage(cfg, l.layers, l.rope, positions)
	if writeStatus(st, nil) != nil {
		return
	}
	shard := Shard{req.Model, req.From, req.To}

	// Forward requests until the coordinator closes the stream. Idle
	// generations are dropped after a while.
	for {
		st.SetReadDeadline(time.Now().Add(5 * time.Minute))
		b, err := readFrame(st, maxRequest)
		if err != nil {
			return
		}
		if json.Unmarshal(b, &req) != nil || req.Op != "forward" {
			st.Reset()
			return
		}
		if req.Tokens <= 0 || req.Pos < 0 || req.Pos+req.Tokens > positions {
			writeStatus(st, fmt.Errorf("positions %d..%d outside the context of %d", req.Pos, req.Pos+req.Tokens, positions))
			return
		}
		acts, err := readFrame(st, 4*req.Tokens*cfg.Dim)
		if err != nil || len(acts) != 4*req.Tokens*cfg.Dim {
			st.Reset()
			return
		}
		if s.failNext != nil && s.failNext(shard) {
			st.Reset() // simulated crash
			return
		}
		settle := func(int64) {}
		gas := StageGas(cfg, req.To-req.From, req.Tokens)
		if s.cfg.Trust != nil {
			if settle, err = s.cfg.Trust.Reserve(c.Subject, gas); err != nil {
				writeStatus(st, err)
				return
			}
		}
		xs := decodeActs(acts)
		for t := range req.Tokens {
			stage.Forward(xs[t*cfg.Dim:(t+1)*cfg.Dim], req.Pos+t)
		}
		settle(gas)
		if s.tamper != nil {
			s.tamper(shard, xs)
		}
		if writeStatus(st, nil) != nil || writeFrame(st, encodeActs(xs)) != nil {
			return
		}
	}
}
