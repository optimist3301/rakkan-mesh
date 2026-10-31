package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
)

// GenerateRequest describes a generation.
type GenerateRequest struct {
	Model       blob.ID
	Prompt      string
	MaxTokens   int
	Temperature float32 // 0: always the most likely token
	Seed        uint64
	// Stages is how many pipeline stages to split the layers into.
	// Default: one per member serving stages, at most one per layer.
	Stages int
	// Providers hold the model's blobs, in addition to this node.
	Providers []peer.ID
}

// Event is one generated token.
type Event struct {
	Token int
	Text  string
}

// StageInfo describes where a range of layers ran.
type StageInfo struct {
	From, To int
	Node     peer.ID
}

// Stats summarizes a generation.
type Stats struct {
	PromptTokens, Tokens int
	Elapsed              time.Duration
	Stages               []StageInfo
	// Failovers counts stages moved to another node mid-generation.
	Failovers int
}

// coordModel is what a coordinator holds of a model: the parts it runs
// itself.
type coordModel struct {
	manifest *Manifest
	embed    []float32
	head     *Head
	tok      *Tokenizer
}

// ErrNoStages means no member can serve the model's layers.
var ErrNoStages = errors.New("no member serves model stages")

func (s *Service) coordinator(ctx context.Context, model blob.ID, providers []peer.ID) (*coordModel, error) {
	s.mu.Lock()
	cm, ok := s.coords[model]
	s.mu.Unlock()
	if ok {
		return cm, nil
	}
	m, err := s.fetchManifest(ctx, model, providers)
	if err != nil {
		return nil, err
	}
	c := m.Config
	part := func(idStr string, max int64) ([]byte, error) {
		id, _ := blob.ParseID(idStr)
		if err := s.blobs.Fetch(ctx, id, providers, max); err != nil {
			return nil, err
		}
		return s.blobs.Store().ReadAll(id, max)
	}
	cm = &coordModel{manifest: m}
	embedSize := int64(4 * c.Vocab * c.Dim)
	data, err := part(m.Embed, embedSize)
	if err != nil {
		return nil, err
	}
	if cm.embed, err = LoadEmbed(c, data); err != nil {
		return nil, err
	}
	if data, err = part(m.Head, embedSize+int64(4*c.Dim)); err != nil {
		return nil, err
	}
	if cm.head, err = LoadHead(c, data, cm.embed); err != nil {
		return nil, err
	}
	if data, err = part(m.Tokenizer, 64<<20); err != nil {
		return nil, err
	}
	if cm.tok, err = ReadTokenizer(data, c.Vocab); err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.coords[model] = cm
	s.mu.Unlock()
	return cm, nil
}

// stageConn runs one stage of a generation, remotely or in this process.
type stageConn interface {
	forward(xs []float32, pos, tokens int) error // in place
	close()
}

type remoteStage struct {
	st network.Stream
	// perToken estimates a forward's duration per token, to tell a dead
	// node from a slow one quickly: a killed process closes no connection,
	// and QUIC would otherwise notice only after its idle timeout.
	perToken time.Duration
}

// deadline allows ten times the usual time per token, plus slack; the
// first forward, a prefill with no estimate yet, gets a generous bound.
func (r *remoteStage) deadline(tokens int) time.Duration {
	if r.perToken == 0 {
		return 2 * time.Minute
	}
	return 2*time.Second + time.Duration(tokens)*10*r.perToken
}

func (r *remoteStage) forward(xs []float32, pos, tokens int) error {
	start := time.Now()
	r.st.SetDeadline(start.Add(r.deadline(tokens)))
	b, _ := json.Marshal(request{Op: "forward", Pos: pos, Tokens: tokens})
	if err := writeFrame(r.st, b); err != nil {
		return err
	}
	if err := writeFrame(r.st, encodeActs(xs)); err != nil {
		return err
	}
	if err := readStatus(r.st); err != nil {
		return err
	}
	out, err := readFrame(r.st, 4*len(xs))
	if err != nil {
		return err
	}
	if len(out) != 4*len(xs) {
		return errors.New("stage returned activations of the wrong size")
	}
	copy(xs, decodeActs(out))
	// Exponentially weighted average, per token.
	sample := time.Since(start) / time.Duration(tokens)
	if r.perToken == 0 {
		r.perToken = sample
	} else {
		r.perToken = (7*r.perToken + sample) / 8
	}
	return nil
}

func (r *remoteStage) close() { r.st.Close() }

type localStage struct {
	stage   *Stage
	dim     int
	release func()
}

func (l *localStage) forward(xs []float32, pos, tokens int) error {
	for t := range tokens {
		l.stage.Forward(xs[t*l.dim:(t+1)*l.dim], pos+t)
	}
	return nil
}

func (l *localStage) close() { l.release() }

// openStage starts a stage on a node: this one, or a member.
func (s *Service) openStage(ctx context.Context, node peer.ID, model blob.ID, from, to int, providers []peer.ID) (stageConn, error) {
	if node == s.node.ID() {
		l, err := s.acquire(ctx, model, from, to, providers)
		if err != nil {
			return nil, err
		}
		c := l.manifest.Config
		cache := CacheBytes(c, to-from)
		if err := s.reserve(cache); err != nil {
			s.release(l)
			return nil, err
		}
		return &localStage{stage: NewStage(c, l.layers, l.rope), dim: c.Dim, release: func() {
			s.unreserve(cache)
			s.release(l)
		}}, nil
	}
	st, err := s.node.NewStream(ctx, node, Protocol)
	if err != nil {
		return nil, err
	}
	// Loading may fetch the layers first.
	st.SetDeadline(time.Now().Add(10 * time.Minute))
	req := request{Op: "open", Model: model.String(), From: from, To: to}
	for _, p := range providers {
		req.Providers = append(req.Providers, p.String())
	}
	b, _ := json.Marshal(req)
	if err := writeFrame(st, b); err != nil {
		st.Reset()
		return nil, err
	}
	if err := readStatus(st); err != nil {
		st.Reset()
		return nil, fmt.Errorf("%s: %w", node.ShortString(), err)
	}
	return &remoteStage{st: st}, nil
}

// planned is a stage of a running generation.
type planned struct {
	from, to int
	node     peer.ID
	conn     stageConn
	inputs   []float32 // activations entering the stage, every position so far
}

// candidates lists nodes able to serve stages, preferring those already
// holding the range.
func (s *Service) candidates(model blob.ID, from, to int, exclude map[peer.ID]bool) []peer.ID {
	key := Shard{model.String(), from, to}
	var loaded, others []peer.ID
	byMem := map[peer.ID]int64{}
	for p, a := range s.servers() {
		if exclude[p] {
			continue
		}
		byMem[p] = a.FreeMem
		if slices.Contains(a.Shards, key) {
			loaded = append(loaded, p)
		} else {
			others = append(others, p)
		}
	}
	if s.cfg.Serve && !exclude[s.node.ID()] {
		others = append(others, s.node.ID())
		byMem[s.node.ID()] = s.cfg.MemoryBytes
	}
	slices.SortFunc(others, func(a, b peer.ID) int { return int(min(max(byMem[b]-byMem[a], -1), 1)) })
	return append(loaded, others...)
}

// Generate runs a generation across the network, calling emit for every
// token. A stage whose node fails is moved to another node and its cache
// rebuilt from the inputs it had received, so the output is exactly what
// an uninterrupted generation would have produced.
func (s *Service) Generate(ctx context.Context, req GenerateRequest, emit func(Event) error) (*Stats, error) {
	started := time.Now()
	providers := append([]peer.ID{s.node.ID()}, req.Providers...)
	for p := range s.servers() {
		providers = append(providers, p)
	}
	cm, err := s.coordinator(ctx, req.Model, providers)
	if err != nil {
		return nil, err
	}
	c := cm.manifest.Config

	n := req.Stages
	if n <= 0 {
		n = len(s.servers())
		if s.cfg.Serve {
			n++
		}
	}
	n = min(max(n, 1), c.Layers)
	stages := make([]*planned, n)
	used := map[peer.ID]bool{}
	defer func() {
		for _, st := range stages {
			if st != nil && st.conn != nil {
				st.conn.close()
			}
		}
	}()
	for i := range n {
		st := &planned{from: i * c.Layers / n, to: (i + 1) * c.Layers / n}
		if err := s.place(ctx, req.Model, st, used, providers); err != nil {
			return nil, err
		}
		used[st.node] = true
		stages[i] = st
	}

	tokens := cm.tok.Encode(req.Prompt, true)
	if len(tokens) >= c.SeqLen {
		return nil, fmt.Errorf("prompt of %d tokens fills the context of %d", len(tokens), c.SeqLen)
	}
	stats := &Stats{PromptTokens: len(tokens)}
	sampler := NewSampler(req.Temperature, req.Seed)
	maxTokens := min(req.MaxTokens, c.SeqLen-len(tokens))
	if req.MaxTokens <= 0 {
		maxTokens = c.SeqLen - len(tokens)
	}

	// Prefill: the whole prompt through each stage in one request.
	xs := make([]float32, 0, len(tokens)*c.Dim)
	for _, t := range tokens {
		xs = append(xs, Embed(c, cm.embed, t)...)
	}
	pos := 0
	for stats.Tokens < maxTokens {
		count := len(xs) / c.Dim
		for _, st := range stages {
			if err := s.runStage(ctx, req.Model, st, xs, pos, count, stages, providers, stats); err != nil {
				return nil, err
			}
		}
		pos += count
		last := xs[(count-1)*c.Dim:]
		next := sampler.Sample(cm.head.Logits(c, last))
		if next == TokenBOS || next == TokenEOS {
			break
		}
		if err := emit(Event{Token: next, Text: cm.tok.Decode(tokens[len(tokens)-1], next)}); err != nil {
			return nil, err
		}
		tokens = append(tokens, next)
		stats.Tokens++
		xs = Embed(c, cm.embed, next)
	}
	stats.Elapsed = time.Since(started)
	for _, st := range stages {
		stats.Stages = append(stats.Stages, StageInfo{From: st.from, To: st.to, Node: st.node})
	}
	return stats, nil
}

// place opens a stage on the best available node.
func (s *Service) place(ctx context.Context, model blob.ID, st *planned, exclude map[peer.ID]bool, providers []peer.ID) error {
	var lastErr error = ErrNoStages
	for _, node := range s.candidates(model, st.from, st.to, exclude) {
		conn, err := s.openStage(ctx, node, model, st.from, st.to, providers)
		if err != nil {
			lastErr = err
			s.log.Info("stage refused", "node", node.ShortString(), "err", err)
			continue
		}
		st.node, st.conn = node, conn
		return nil
	}
	return fmt.Errorf("layers %d..%d: %w", st.from, st.to, lastErr)
}

// runStage forwards activations through a stage, moving it to another
// node if its node fails: the replacement's cache is rebuilt by replaying
// the inputs the stage received so far, which the coordinator keeps.
func (s *Service) runStage(ctx context.Context, model blob.ID, st *planned, xs []float32, pos, count int, all []*planned, providers []peer.ID, stats *Stats) error {
	in := slices.Clone(xs)
	for attempt := 0; ; attempt++ {
		err := st.conn.forward(xs, pos, count)
		if err == nil {
			st.inputs = append(st.inputs, in...)
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if attempt >= 3 {
			return fmt.Errorf("layers %d..%d: %w", st.from, st.to, err)
		}
		s.log.Warn("stage failed, moving it", "node", st.node.ShortString(), "from", st.from, "to", st.to, "err", err)
		st.conn.close()
		exclude := map[peer.ID]bool{st.node: true}
		for _, o := range all {
			if o != st {
				exclude[o.node] = true
			}
		}
		if err := s.place(ctx, model, st, exclude, providers); err != nil {
			// Doubling up on a node already in the pipeline beats failing.
			if err = s.place(ctx, model, st, map[peer.ID]bool{st.node: true}, providers); err != nil {
				return err
			}
		}
		if pos > 0 {
			replay := slices.Clone(st.inputs)
			if err := st.conn.forward(replay, 0, pos); err != nil {
				continue
			}
		}
		stats.Failovers++
		copy(xs, in)
	}
}
