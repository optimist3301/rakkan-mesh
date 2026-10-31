package llm

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
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
	// Chat wraps the prompt in the model's conversation format, for
	// chat-tuned models; it is an error for models without one.
	Chat bool
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
	// Witness is the trusted node checking a stage on a guest, if any.
	Witness peer.ID
}

// Stats summarizes a generation.
type Stats struct {
	PromptTokens, Tokens int
	Elapsed              time.Duration
	Stages               []StageInfo
	// Failovers counts stages moved to another node mid-generation.
	Failovers int
	// Caught counts stages whose node returned outputs differing from its
	// witness's; the witness took over each.
	Caught int
}

// coordModel is what a coordinator holds of a model: the parts it runs
// itself.
type coordModel struct {
	manifest *Manifest
	embed    Matrix
	head     *Head
	tok      Codec
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
	embedSize := EmbedBytes(c)
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
	if cm.tok, err = readCodec(m, data); err != nil {
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
func (s *Service) openStage(ctx context.Context, node peer.ID, model blob.ID, from, to, positions int, providers []peer.ID) (stageConn, error) {
	if node == s.node.ID() {
		l, err := s.acquire(ctx, model, from, to, providers)
		if err != nil {
			return nil, err
		}
		c := l.manifest.Config
		cache := CacheBytes(c, to-from, positions)
		if err := s.reserve(cache); err != nil {
			s.release(l)
			return nil, err
		}
		return &localStage{stage: NewStage(c, l.layers, l.rope, positions), dim: c.Dim, release: func() {
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
	req := request{Op: "open", Model: model.String(), From: from, To: to, Ctx: positions}
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
	// positions is the generation's length: prompt and new tokens.
	positions int
	node      peer.ID
	conn      stageConn
	witness   *witness
	// unwitnessed is set when no trusted node could witness the stage on
	// its current node; it is not looked for again until the stage moves.
	unwitnessed bool
	inputs      []float32 // activations entering the stage, every position so far
}

// witness runs a stage placed on an untrusted guest a second time, on a
// trusted node: the engine is deterministic, so an honest guest's outputs
// match the witness's bit for bit.
type witness struct {
	node peer.ID
	conn stageConn
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
	byFree := func(a, b peer.ID) int {
		if c := cmp.Compare(byMem[b], byMem[a]); c != 0 {
			return c
		}
		return cmp.Compare(a, b)
	}
	slices.SortFunc(loaded, byFree)
	slices.SortFunc(others, byFree)
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

	m := cm.manifest
	prompt := req.Prompt
	if req.Chat {
		if m.Chat == nil {
			return nil, errors.New("the model has no chat format")
		}
		prompt = m.Chat.Prefix + prompt + m.Chat.Suffix
	}
	tokens := cm.tok.Encode(prompt)
	if bos := m.bos(); bos >= 0 {
		tokens = append([]int{bos}, tokens...)
	}
	if len(tokens) == 0 {
		return nil, errors.New("empty prompt")
	}
	if len(tokens) >= c.SeqLen {
		return nil, fmt.Errorf("prompt of %d tokens fills the context of %d", len(tokens), c.SeqLen)
	}
	stats := &Stats{PromptTokens: len(tokens)}
	sampler := NewSampler(req.Temperature, req.Seed)
	maxTokens := c.SeqLen - len(tokens)
	if req.MaxTokens > 0 {
		maxTokens = min(req.MaxTokens, maxTokens)
	}
	// Stages keep caches for exactly this many positions.
	positions := len(tokens) + maxTokens

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
			if st != nil && st.witness != nil {
				st.witness.conn.close()
			}
		}
	}()
	for i := range n {
		st := &planned{from: i * c.Layers / n, to: (i + 1) * c.Layers / n, positions: positions}
		if err := s.place(ctx, req.Model, st, used, providers); err != nil {
			return nil, err
		}
		used[st.node] = true
		stages[i] = st
	}

	// Prefill: the whole prompt through each stage in one request.
	xs := make([]float32, 0, len(tokens)*c.Dim)
	for _, t := range tokens {
		xs = append(xs, Embed(c, cm.embed, t)...)
	}
	stop := m.stop()
	var pending []byte // bytes of a character split across tokens
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
		if slices.Contains(stop, next) {
			break
		}
		pending = append(pending, cm.tok.Piece(tokens[len(tokens)-1], next)...)
		var text []byte
		text, pending = validUTF8Prefix(pending)
		if err := emit(Event{Token: next, Text: string(text)}); err != nil {
			return nil, err
		}
		pending = slices.Clone(pending)
		tokens = append(tokens, next)
		stats.Tokens++
		xs = Embed(c, cm.embed, next)
	}
	stats.Elapsed = time.Since(started)
	for _, st := range stages {
		info := StageInfo{From: st.from, To: st.to, Node: st.node}
		if st.witness != nil {
			info.Witness = st.witness.node
		}
		stats.Stages = append(stats.Stages, info)
	}
	return stats, nil
}

// place opens a stage on the best available node.
func (s *Service) place(ctx context.Context, model blob.ID, st *planned, exclude map[peer.ID]bool, providers []peer.ID) error {
	var lastErr error = ErrNoStages
	for _, node := range s.candidates(model, st.from, st.to, exclude) {
		if s.cfg.Trust != nil && s.cfg.Trust.Suspect(node) {
			continue
		}
		conn, err := s.openStage(ctx, node, model, st.from, st.to, st.positions, providers)
		if err != nil {
			lastErr = err
			s.log.Info("stage refused", "node", node.ShortString(), "err", err)
			continue
		}
		st.node, st.conn, st.unwitnessed = node, conn, false
		return nil
	}
	return fmt.Errorf("layers %d..%d: %w", st.from, st.to, lastErr)
}

// runStage forwards activations through a stage, moving it to another
// node if its node fails: the replacement's cache is rebuilt by replaying
// the inputs the stage received so far, which the coordinator keeps. A
// stage on an untrusted guest runs on a witness too, and the outputs are
// compared; on a mismatch the witness takes the stage over and the guest
// is reported as cheating.
func (s *Service) runStage(ctx context.Context, model blob.ID, st *planned, xs []float32, pos, count int, all []*planned, providers []peer.ID, stats *Stats) error {
	in := slices.Clone(xs)
	gas := StageGas(s.manifestConfig(model), st.to-st.from, count)
	for attempt := 0; ; attempt++ {
		s.ensureWitness(ctx, model, st, pos, providers)
		var check []float32
		var checkErr error
		var wg sync.WaitGroup
		w := st.witness
		if w != nil {
			check = slices.Clone(in)
			wg.Go(func() { checkErr = w.conn.forward(check, pos, count) })
		}
		err := st.conn.forward(xs, pos, count)
		wg.Wait()
		if w != nil {
			switch {
			case checkErr != nil:
				// A new witness replays the inputs on the next forward.
				s.log.Warn("stage witness failed", "node", w.node.ShortString(), "err", checkErr)
				w.conn.close()
				st.witness = nil
			case err != nil:
				// The witness has this step done: it takes over.
				s.log.Warn("stage failed, its witness takes over", "node", st.node.ShortString(), "err", err)
				st.conn.close()
				st.node, st.conn, st.witness = w.node, w.conn, nil
				copy(xs, check)
				stats.Failovers++
				err = nil
			case !bytes.Equal(encodeActs(xs), encodeActs(check)):
				s.cfg.Trust.Cheated(st.node, fmt.Sprintf("layers %d..%d: outputs differ from a trusted witness", st.from, st.to))
				st.conn.close()
				st.node, st.conn, st.witness = w.node, w.conn, nil
				copy(xs, check)
				stats.Caught++
			default:
				s.cfg.Trust.Verified(st.node)
				s.cfg.Trust.Consumed(w.node, gas)
			}
		}
		if err == nil {
			if s.cfg.Trust != nil {
				s.cfg.Trust.Consumed(st.node, gas)
			}
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
		if s.cfg.Trust != nil {
			s.cfg.Trust.Failed(st.node)
		}
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

// ensureWitness gives a stage on an untrusted guest a witness on a trusted
// node, replaying the stage's inputs so far into it. Without a trusted node
// able to serve the layers, the stage runs unchecked.
func (s *Service) ensureWitness(ctx context.Context, model blob.ID, st *planned, pos int, providers []peer.ID) {
	tr := s.cfg.Trust
	if tr == nil || st.witness != nil || st.unwitnessed || tr.Trusted(st.node) {
		if st.witness != nil && tr.Trusted(st.node) {
			st.witness.conn.close()
			st.witness = nil
		}
		return
	}
	for _, node := range s.candidates(model, st.from, st.to, map[peer.ID]bool{st.node: true}) {
		if !tr.Trusted(node) {
			continue
		}
		conn, err := s.openStage(ctx, node, model, st.from, st.to, st.positions, providers)
		if err != nil {
			continue
		}
		if pos > 0 {
			if err := conn.forward(slices.Clone(st.inputs), 0, pos); err != nil {
				conn.close()
				continue
			}
		}
		st.witness = &witness{node: node, conn: conn}
		return
	}
	st.unwitnessed = true
	s.log.Warn("no trusted node to witness a guest's stage; it runs unchecked", "node", st.node.ShortString(), "from", st.from, "to", st.to)
}

// manifestConfig returns the configuration of a model being coordinated.
func (s *Service) manifestConfig(model blob.ID) Config {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.coords[model].manifest.Config
}
