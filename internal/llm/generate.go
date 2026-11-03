package llm

import (
	"bytes"
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
	// A llama model's embedding, head and tokenizer.
	embed Matrix
	head  *Head
	tok   Codec
	// A DeepSeek model's.
	ds *deepseekCoord
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
	cm = &coordModel{manifest: m}
	if m.ds != nil {
		if cm.ds, err = s.deepseekCoordinator(ctx, m, providers); err != nil {
			return nil, err
		}
	} else if err := s.llamaCoordinator(ctx, cm, providers); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if prev, ok := s.coords[model]; ok {
		// A concurrent generation loaded it first: keep one copy.
		if cm.head != nil && cm.head.Wcls.dev != nil {
			cm.head.Wcls.dev.Free()
		}
		return prev, nil
	}
	s.coords[model] = cm
	return cm, nil
}

func (s *Service) llamaCoordinator(ctx context.Context, cm *coordModel, providers []peer.ID) error {
	m := cm.manifest
	c := m.Config
	part := func(idStr string, max int64) ([]byte, error) {
		id, _ := blob.ParseID(idStr)
		if err := s.blobs.Fetch(ctx, id, providers, max); err != nil {
			return nil, err
		}
		return s.blobs.Store().ReadAll(id, max)
	}
	embedSize := EmbedBytes(c)
	data, err := part(m.Embed, embedSize)
	if err != nil {
		return err
	}
	if cm.embed, err = LoadEmbed(c, data); err != nil {
		return err
	}
	if data, err = part(m.Head, embedSize+int64(4*c.Dim)); err != nil {
		return err
	}
	if cm.head, err = LoadHead(c, data, cm.embed); err != nil {
		return err
	}
	if d := s.cfg.Device; d != nil {
		// The classifier is the coordinator's largest product. The host copy
		// stays: it is the embedding table too, and the fallback.
		if err := cm.head.Wcls.toDevice(d, c.Vocab, c.Dim, true); err != nil {
			s.log.Warn("classifier stays on the CPU", "err", err)
		}
	}
	if data, err = part(m.Tokenizer, 64<<20); err != nil {
		return err
	}
	cm.tok, err = readCodec(m, data)
	return err
}

// prompt encodes a generation's prompt.
func (cm *coordModel) prompt(req GenerateRequest) ([]int, error) {
	if cm.ds != nil {
		return cm.ds.prompt(req.Prompt, req.Chat)
	}
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
	return tokens, nil
}

func (cm *coordModel) stop() []int {
	if cm.ds != nil {
		return []int{cm.ds.head.Cfg.EOS}
	}
	return cm.manifest.stop()
}

func (cm *coordModel) piece(prev, next int) []byte {
	if cm.ds != nil {
		return cm.ds.tok.Piece(prev, next)
	}
	return cm.tok.Piece(prev, next)
}

// inputs are the first stage's input for positions from.. of tokens, the
// prompt being their first prompt.
func (cm *coordModel) inputs(tokens []int, from, prompt int) ([]byte, error) {
	if cm.ds != nil {
		return cm.ds.inputs(tokens, from, prompt)
	}
	c := cm.manifest.Config
	xs := make([]float32, 0, (len(tokens)-from)*c.Dim)
	for _, t := range tokens[from:] {
		xs = append(xs, Embed(c, cm.embed, t)...)
	}
	return encodeActs(xs), nil
}

// logits are the next token's scores, from the last stage's output for
// count positions.
func (cm *coordModel) logits(out []byte, count, positions int) ([]float32, error) {
	if cm.ds != nil {
		return cm.ds.logits(out, count, positions)
	}
	c := cm.manifest.Config
	if len(out) != 4*count*c.Dim {
		return nil, errors.New("stage returned activations of the wrong size")
	}
	last := decodeActs(out[4*(count-1)*c.Dim:])
	return cm.head.Logits(c, last), nil
}

// stageConn runs one stage of a generation, remotely or in this process,
// on what passes between stages: activations for a llama model, dsv41's
// Payloads for DeepSeek.
type stageConn interface {
	forward(in []byte, pos, tokens int) ([]byte, error)
	close()
}

type remoteStage struct {
	st    network.Stream
	shape shape
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

func (r *remoteStage) forward(in []byte, pos, tokens int) ([]byte, error) {
	start := time.Now()
	r.st.SetDeadline(start.Add(r.deadline(tokens)))
	b, _ := json.Marshal(request{Op: "forward", Pos: pos, Tokens: tokens})
	if err := writeFrame(r.st, b); err != nil {
		return nil, err
	}
	if err := writeFrame(r.st, in); err != nil {
		return nil, err
	}
	if err := readStatus(r.st); err != nil {
		return nil, err
	}
	out, err := readFrame(r.st, r.shape.payloadBytes(tokens))
	if err != nil {
		return nil, err
	}
	// Exponentially weighted average, per token.
	sample := time.Since(start) / time.Duration(tokens)
	if r.perToken == 0 {
		r.perToken = sample
	} else {
		r.perToken = (7*r.perToken + sample) / 8
	}
	return out, nil
}

func (r *remoteStage) close() { r.st.Close() }

type localStage struct {
	stageConn
	release func()
}

func (l *localStage) close() {
	l.stageConn.close()
	l.release()
}

// openStage starts a stage on a node: this one, or a member.
func (s *Service) openStage(ctx context.Context, node peer.ID, model blob.ID, from, to, positions int, providers []peer.ID) (stageConn, error) {
	if node == s.node.ID() {
		l, err := s.acquire(ctx, model, from, to, providers)
		if err != nil {
			return nil, err
		}
		unreserve, err := s.reserveCache(l, from, to, positions)
		if err != nil {
			s.release(l)
			return nil, err
		}
		return &localStage{stageConn: l.open(positions), release: func() {
			unreserve()
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
	return &remoteStage{st: st, shape: s.manifestShape(model)}, nil
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
	inputs      []byte // what entered the stage, every position so far
}

// witness runs a stage placed on an untrusted guest a second time, on a
// trusted node: the engine is deterministic, so an honest guest's outputs
// match the witness's bit for bit.
type witness struct {
	node peer.ID
	conn stageConn
}

// candidates lists nodes able to hold layers from..to for a generation of
// positions, those already holding the range first, then the fastest.
func (s *Service) candidates(model blob.ID, from, to, positions int, exclude map[peer.ID]bool) []peer.ID {
	key := Shard{model.String(), from, to}
	sh := s.manifestShape(model)
	servers := s.servers()
	var nodes []stageNode
	for _, n := range s.stageNodes(model.String(), sh, positions, exclude) {
		if n.fits(from, to) {
			nodes = append(nodes, n)
		}
	}
	sortNodes(nodes)
	var loaded, others []peer.ID
	for _, n := range nodes {
		if slices.Contains(servers[n.id].Shards, key) {
			loaded = append(loaded, n.id)
		} else {
			others = append(others, n.id)
		}
	}
	return append(loaded, others...)
}

// deepseekMaxTokens is a DeepSeek generation's default length: its
// context of a million positions would have every stage reserve caches
// for all of them.
const deepseekMaxTokens = 4096

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
	m := cm.manifest
	sh := m.shape()
	layers, seqLen := sh.numLayers(), m.seqLen()

	tokens, err := cm.prompt(req)
	if err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, errors.New("empty prompt")
	}
	if len(tokens) >= seqLen {
		return nil, fmt.Errorf("prompt of %d tokens fills the context of %d", len(tokens), seqLen)
	}
	stats := &Stats{PromptTokens: len(tokens)}
	sampler := NewSampler(req.Temperature, req.Seed)
	maxTokens := seqLen - len(tokens)
	if req.MaxTokens > 0 {
		maxTokens = min(req.MaxTokens, maxTokens)
	} else if m.ds != nil {
		maxTokens = min(deepseekMaxTokens, maxTokens)
	}
	// Stages keep caches for exactly this many positions.
	positions := len(tokens) + maxTokens

	// Assign layers by the nodes' speed and memory.
	cost := make([]int64, layers)
	for i := range cost {
		cost[i] = sh.readBytes(i)
	}
	spans, err := planStages(cost, s.stageNodes(req.Model.String(), sh, positions, nil), min(req.Stages, layers))
	if err != nil {
		return nil, err
	}
	stages := make([]*planned, len(spans))
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
	for i, sp := range spans {
		st := &planned{from: sp.from, to: sp.to, positions: positions}
		// The planned node, or, if it refuses, the next best.
		if conn, err := s.openStage(ctx, sp.node, req.Model, sp.from, sp.to, positions, providers); err == nil {
			st.node, st.conn = sp.node, conn
		} else {
			s.log.Info("stage refused", "node", sp.node.ShortString(), "err", err)
			if err := s.place(ctx, req.Model, st, used, providers); err != nil {
				return nil, err
			}
		}
		used[st.node] = true
		stages[i] = st
	}

	// Prefill: the whole prompt through each stage in one request.
	prompt := len(tokens)
	in, err := cm.inputs(tokens, 0, prompt)
	if err != nil {
		return nil, err
	}
	stop := cm.stop()
	var pending []byte // bytes of a character split across tokens
	pos := 0
	for stats.Tokens < maxTokens {
		count := len(tokens) - pos
		for _, st := range stages {
			if in, err = s.runStage(ctx, req.Model, st, in, pos, count, stages, providers, stats); err != nil {
				return nil, err
			}
		}
		logits, err := cm.logits(in, count, positions)
		if err != nil {
			return nil, err
		}
		pos += count
		next := sampler.Sample(logits)
		if slices.Contains(stop, next) {
			break
		}
		pending = append(pending, cm.piece(tokens[len(tokens)-1], next)...)
		var text []byte
		text, pending = validUTF8Prefix(pending)
		if err := emit(Event{Token: next, Text: string(text)}); err != nil {
			return nil, err
		}
		pending = slices.Clone(pending)
		tokens = append(tokens, next)
		stats.Tokens++
		if in, err = cm.inputs(tokens, pos, prompt); err != nil {
			return nil, err
		}
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
	for _, node := range s.candidates(model, st.from, st.to, st.positions, exclude) {
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

// runStage forwards a stage's input through it and returns its output,
// moving the stage to another node if its node fails: the replacement's
// cache is rebuilt by replaying the inputs the stage received so far,
// which the coordinator keeps. A stage on an untrusted guest runs on a
// witness too, and the outputs are compared byte for byte; on a mismatch
// the witness takes the stage over and the guest is reported as cheating.
func (s *Service) runStage(ctx context.Context, model blob.ID, st *planned, in []byte, pos, count int, all []*planned, providers []peer.ID, stats *Stats) ([]byte, error) {
	gas := s.manifestShape(model).gas(st.from, st.to, count)
	for attempt := 0; ; attempt++ {
		s.ensureWitness(ctx, model, st, pos, providers)
		var check []byte
		var checkErr error
		var wg sync.WaitGroup
		w := st.witness
		if w != nil {
			wg.Go(func() { check, checkErr = w.conn.forward(in, pos, count) })
		}
		out, err := st.conn.forward(in, pos, count)
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
				out, err = check, nil
				stats.Failovers++
			case !bytes.Equal(out, check):
				s.cfg.Trust.Cheated(st.node, fmt.Sprintf("layers %d..%d: outputs differ from a trusted witness", st.from, st.to))
				st.conn.close()
				st.node, st.conn, st.witness = w.node, w.conn, nil
				out = check
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
			return out, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if attempt >= 3 {
			return nil, fmt.Errorf("layers %d..%d: %w", st.from, st.to, err)
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
				return nil, err
			}
		}
		if pos > 0 {
			if _, err := st.conn.forward(st.inputs, 0, pos); err != nil {
				continue
			}
		}
		stats.Failovers++
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
	for _, node := range s.candidates(model, st.from, st.to, st.positions, map[peer.ID]bool{st.node: true}) {
		if !tr.Trusted(node) {
			continue
		}
		conn, err := s.openStage(ctx, node, model, st.from, st.to, st.positions, providers)
		if err != nil {
			continue
		}
		if pos > 0 {
			if _, err := conn.forward(st.inputs, 0, pos); err != nil {
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

// manifestShape returns the shape of a model being coordinated.
func (s *Service) manifestShape(model blob.ID) shape {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.coords[model].manifest.shape()
}
