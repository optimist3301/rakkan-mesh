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
	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
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
	// Draft has the model's draft model (DeepSeek-V4.1's DSpark) propose
	// tokens, verified a block at a time: fewer passes through the stages,
	// the same text.
	Draft bool
	// Images are encoded images (PNG) for a model with an image encoder
	// (DeepSeek-V4.1's), in the order of their places: with Chat they come
	// before the text of the message; without, the text marks each place
	// with the model's image placeholder.
	Images [][]byte
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
	// Drafted and Accepted count the draft's tokens proposed and taken.
	Drafted, Accepted int
	// Passes counts passes through the stages.
	Passes int
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
		return cm.ds.prompt(req.Prompt, req.Chat, len(req.Images))
	}
	if len(req.Images) > 0 {
		return nil, errors.New("the model takes no images")
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
// prompt being their first prompt; wantMain asks for what a draft model
// reads (DeepSeek's).
func (cm *coordModel) inputs(tokens []int, rows map[int][]float32, from, prompt int, wantMain bool) ([]byte, error) {
	if cm.ds != nil {
		return cm.ds.inputs(tokens, rows, from, prompt, wantMain)
	}
	c := cm.manifest.Config
	xs := make([]float32, 0, (len(tokens)-from)*c.Dim)
	for _, t := range tokens[from:] {
		xs = append(xs, Embed(c, cm.embed, t)...)
	}
	return encodeActs(xs), nil
}

// outputs reads the last stage's output for count positions: the next
// token's scores after position j, and what a draft model reads of it.
func (cm *coordModel) outputs(out []byte, count, positions int) (logits, main func(j int) []float32, err error) {
	if cm.ds != nil {
		return cm.ds.outputs(out, count, positions)
	}
	c := cm.manifest.Config
	if len(out) != 4*count*c.Dim {
		return nil, nil, errors.New("stage returned activations of the wrong size")
	}
	return func(j int) []float32 { return cm.head.Logits(c, decodeActs(out[4*j*c.Dim:4*(j+1)*c.Dim])) }, func(int) []float32 { return nil }, nil
}

// logits are the scores after the last of count positions.
func (cm *coordModel) logits(out []byte, count, positions int) ([]float32, error) {
	logits, _, err := cm.outputs(out, count, positions)
	if err != nil {
		return nil, err
	}
	return logits(count - 1), nil
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
		conn, err := l.open(positions)
		if err != nil {
			unreserve()
			s.release(l)
			return nil, err
		}
		return &localStage{stageConn: conn, release: func() {
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
	ends        []int  // where each position's input ends in inputs
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
	var rows map[int][]float32 // the inputs of image positions
	if cm.ds != nil {
		if tokens, rows, err = s.deepseekImages(ctx, m, cm.ds, providers, tokens, req.Images); err != nil {
			return nil, err
		}
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

	// A draft model proposes the tokens after each, which a pass of the
	// stages verifies at once: the main model's token after each position is
	// sampled as without the draft, in the same order, and a drafted token
	// is taken when it is that token. The text is the same as without the
	// draft, whatever the temperature; what the draft gets wrong costs only
	// the stages a rewind. The draft reads the main model's stream at its
	// target layers, which the stages report (Payload.Main).
	var draft *dsv41.DraftState
	if req.Draft {
		if cm.ds == nil {
			return nil, errors.New("the model has no draft model")
		}
		d, err := s.deepseekDraft(ctx, m, cm.ds, providers)
		if err != nil {
			return nil, fmt.Errorf("draft model: %w", err)
		}
		draft = d.NewState()
	}

	// Prefill: the whole prompt through each stage in one request.
	prompt := len(tokens)
	in, err := cm.inputs(tokens, rows, 0, prompt, draft != nil)
	if err != nil {
		return nil, err
	}
	stop := cm.stop()
	var pending []byte // bytes of a character split across tokens
	pos := 0
	var proposal []int // drafted tokens after the last of tokens, in in
	done := false
	for !done && stats.Tokens < maxTokens {
		start, count := pos, len(tokens)+len(proposal)-pos
		stats.Passes++
		for _, st := range stages {
			if in, err = s.runStage(ctx, req.Model, st, in, pos, count, stages, providers, stats); err != nil {
				return nil, err
			}
		}
		logits, mains, err := cm.outputs(in, count, positions)
		if err != nil {
			return nil, err
		}
		// The tokens after the last known one, then after each drafted one
		// as long as the draft got it right.
		last := len(tokens) - 1
		for i := 0; ; i++ {
			next := sampler.Sample(logits(last + i - start))
			if slices.Contains(stop, next) {
				done = true
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
			if i == len(proposal) || next != proposal[i] || stats.Tokens >= maxTokens {
				break
			}
			stats.Accepted++
		}
		// Every position run whose input was right, the last token's not yet.
		pos = len(tokens) - 1
		if draft != nil {
			for q := draft.Main(); q < pos; q++ {
				if err := draft.AddMain(mains(q - start)); err != nil {
					return nil, err
				}
			}
		}
		proposal = nil
		if draft != nil && !done && stats.Tokens < maxTokens && tokens[len(tokens)-1] >= 0 {
			// As many as fit the generation's positions and its length.
			k := min(m.ds.DraftBlock, maxTokens-stats.Tokens-1, positions-len(tokens))
			if k > 0 {
				p, err := draft.Propose(tokens[len(tokens)-1], cm.ds.head)
				if err != nil {
					return nil, err
				}
				proposal = p.Tokens[1 : 1+k]
				if s.redraft != nil {
					proposal = s.redraft(tokens, proposal)
				}
				stats.Drafted += len(proposal)
			}
		}
		if in, err = cm.inputs(append(slices.Clone(tokens), proposal...), rows, pos, prompt, draft != nil); err != nil {
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
	sh := s.manifestShape(model)
	gas := sh.gas(st.from, st.to, count)
	// A pass from an earlier position rewinds the stage: what it received
	// for the positions from there on goes.
	if pos < len(st.ends) {
		if pos == 0 {
			st.inputs, st.ends = st.inputs[:0], st.ends[:0]
		} else {
			st.inputs, st.ends = st.inputs[:st.ends[pos-1]], st.ends[:pos]
		}
	}
	if pos != len(st.ends) {
		return nil, fmt.Errorf("layers %d..%d: position %d after %d", st.from, st.to, pos, len(st.ends))
	}
	sizes, err := sh.sizes(in, count)
	if err != nil {
		return nil, err
	}
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
			for _, n := range sizes {
				st.ends = append(st.ends, len(st.inputs)+n)
				st.inputs = append(st.inputs, in[:n]...)
				in = in[n:]
			}
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
