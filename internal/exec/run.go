package exec

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

var (
	ErrNotEnoughNodes = errors.New("not enough distinct members for the requested replicas")
	ErrNoConsensus    = errors.New("replicas disagree")
)

// Request is a task to run somewhere in the network.
type Request struct {
	Module []byte
	Entry  string
	// Input, or InputBlob for data already in the local blob store. Inputs
	// larger than 1 MiB are stored as blobs automatically and fetched by
	// the executor in verified chunks.
	Input     []byte
	InputBlob blob.ID
	// Providers are other members holding InputBlob, which executors may
	// fetch it from in parallel with this node.
	Providers []peer.ID
	Limits    sandbox.Limits
	// Replicas > 1 runs the task on that many distinct members and accepts
	// the outcome a majority agrees on.
	Replicas int
}

// Result is a task's outcome and where it ran.
type Result struct {
	sandbox.Result
	// OutputBlob names an output larger than 1 MiB, fetched into the local
	// blob store; Output is then empty.
	OutputBlob blob.ID
	Node       peer.ID
	Attempts   int
	// Replicas lists every replica's node and whether it agreed with the
	// accepted outcome, when the task was replicated.
	Replicas []Replica
}

type Replica struct {
	Node   peer.ID
	Agreed bool
	Err    error
}

// Run executes a task on the best available member.
func (s *Service) Run(ctx context.Context, req Request) (*Result, error) {
	if req.Limits == (sandbox.Limits{}) {
		req.Limits = sandbox.DefaultLimits()
	}
	req, unpin, err := s.prepareInput(req)
	if err != nil {
		return nil, err
	}
	defer unpin()
	if req.Replicas > 1 {
		return s.runReplicated(ctx, req)
	}
	res, err := s.runOnce(ctx, req, newPlacement())
	if err != nil {
		return nil, err
	}
	return res, nil
}

// prepareInput turns a large inline input into a blob, so retries and
// replicas share one stored copy and executors fetch it in verified chunks,
// and pins the input blob while the task runs.
func (s *Service) prepareInput(req Request) (Request, func(), error) {
	store := s.cfg.Blobs.Store()
	if req.InputBlob.IsZero() && len(req.Input) > inlineMax {
		id, err := store.PutBytes(req.Input)
		if err != nil {
			return req, nil, err
		}
		req.InputBlob, req.Input = id, nil
	}
	if req.InputBlob.IsZero() {
		return req, func() {}, nil
	}
	unpin, ok := store.Pin(req.InputBlob)
	if !ok {
		return req, nil, fmt.Errorf("%w: input blob %s", blob.ErrNotFound, req.InputBlob)
	}
	return req, unpin, nil
}

// placement tracks executors already used by a task, so retries and
// replicas go to different ones.
type placement struct {
	mu   sync.Mutex
	used map[peer.ID]bool
}

func newPlacement() *placement { return &placement{used: map[peer.ID]bool{}} }

// runOnce runs a task on one executor, retrying on others when the
// executor fails. Busy executors are waited out without counting as an
// attempt; the context bounds the wait.
func (s *Service) runOnce(ctx context.Context, req Request, pl *placement) (*Result, error) {
	hash := HashOf(req.Module)
	var lastErr error
	for attempt := 1; attempt <= s.cfg.MaxAttempts; {
		target, anyCandidate, limitErr := s.pick(hash, req.Limits, pl)
		if target == "" {
			if !anyCandidate {
				switch {
				case lastErr != nil:
					return nil, fmt.Errorf("%w (last attempt: %v)", ErrNoCapacity, lastErr)
				case limitErr != nil:
					return nil, fmt.Errorf("%w: the task's limits exceed every executor's policy (%v)", ErrNoCapacity, limitErr)
				}
				return nil, ErrNoCapacity
			}
			// Everyone able to run it is full: wait for capacity.
			if err := s.waitCapacity(ctx); err != nil {
				return nil, err
			}
			continue
		}
		res, err := s.execOn(ctx, target, hash, req)
		if err == nil {
			res.Attempts = attempt
			return res, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		switch {
		case errors.Is(err, ErrBusy):
			s.markFull(target)
			s.release1(pl, target) // it may have room again later
			continue
		case !retryable(err):
			return nil, err
		}
		s.log.Info("executor failed, retrying elsewhere", "peer", target.ShortString(), "err", err)
		lastErr = err
		attempt++
	}
	return nil, fmt.Errorf("task failed on %d executors: %w", s.cfg.MaxAttempts, lastErr)
}

// retryable reports whether an error is the executor's fault rather than
// the task's: another executor would likely succeed.
func retryable(err error) bool {
	var re *RemoteError
	if errors.As(err, &re) {
		switch re.Kind {
		// "canceled" here means the executor gave up on its own (shutting
		// down): our own cancellations never reach the wire.
		case "worker_crashed", "worker_protocol", "internal", "canceled":
			return true
		}
		return false
	}
	if errors.Is(err, ErrRejected) {
		return true // another node may have a higher ceiling
	}
	// Local sandbox errors are deterministic; anything else is transport.
	return ErrorKind(err) == "internal" || ErrorKind(err) == "worker_crashed" || ErrorKind(err) == "worker_protocol"
}

func (s *Service) waitCapacity(ctx context.Context) error {
	t := time.NewTimer(50*time.Millisecond + rand.N(100*time.Millisecond))
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// markFull assumes a busy executor is full until it gossips again.
func (s *Service) markFull(p peer.ID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if l, ok := s.loads[p]; ok {
		l.Busy, l.Queued = l.Slots, l.Slots
		s.loads[p] = l
	}
}

func (s *Service) release1(pl *placement, p peer.ID) {
	pl.mu.Lock()
	delete(pl.used, p)
	pl.mu.Unlock()
}

// pick chooses and reserves an executor: most free capacity first, then
// one already holding the module, then the lowest round-trip time. It
// returns "" when nobody has room; anyCandidate tells whether anybody
// could run the task at all, and limitErr why an executor's policy refused
// the task's limits, if one did.
func (s *Service) pick(hash Hash, lim sandbox.Limits, pl *placement) (target peer.ID, anyCandidate bool, limitErr error) {
	type cand struct {
		id    peer.ID
		score float64
		rtt   time.Duration
	}
	var cands []cand
	self := s.node.ID()
	pl.mu.Lock()
	defer pl.mu.Unlock()

	if s.cfg.Pool != nil && !pl.used[self] {
		limitErr = lim.Within(s.cfg.Ceiling)
	}
	if s.cfg.Pool != nil && !pl.used[self] && limitErr == nil {
		anyCandidate = true
		free := s.cfg.Slots + s.cfg.MaxQueue - int(s.inflight.Load())
		if free > 0 {
			// Prefer running locally on ties: no transfer.
			cands = append(cands, cand{id: self, score: float64(min(free, s.cfg.Slots)) + 0.75})
		}
	}
	members := s.node.Members()
	s.mu.Lock()
	for _, m := range members {
		l, ok := s.loads[m.ID]
		if !m.Connected || !ok || !s.fresh(l) || l.Slots == 0 || pl.used[m.ID] {
			continue
		}
		if err := lim.Within(l.Ceiling); err != nil {
			limitErr = err
			continue
		}
		anyCandidate = true
		free := l.Slots - l.Busy - l.Queued - s.pending[m.ID]
		if free <= 0 {
			continue
		}
		score := float64(free)
		if s.placed[m.ID][hash] {
			score += 0.5
		}
		cands = append(cands, cand{id: m.ID, score: score, rtt: m.RTT})
	}
	s.mu.Unlock()
	if len(cands) == 0 {
		return "", anyCandidate, limitErr
	}
	best := cands[0]
	for _, c := range cands[1:] {
		if c.score > best.score || (c.score == best.score && c.rtt < best.rtt) {
			best = c
		}
	}
	pl.used[best.id] = true
	return best.id, true, nil
}

// execOn runs a task on a given executor.
func (s *Service) execOn(ctx context.Context, target peer.ID, hash Hash, req Request) (*Result, error) {
	if target == s.node.ID() {
		if !s.admit() {
			return nil, ErrBusy
		}
		defer s.release()
		input := req.Input
		if !req.InputBlob.IsZero() {
			var err error
			if input, err = s.cfg.Blobs.Store().ReadAll(req.InputBlob, int64(req.Limits.MaxInputBytes)); err != nil {
				if errors.Is(err, blob.ErrTooLarge) {
					return nil, fmt.Errorf("%w: %v", sandbox.ErrInputTooLarge, err)
				}
				return nil, err
			}
		}
		res, err := s.cfg.Pool.Run(ctx, string(target), sandbox.Task{Module: req.Module, Entry: req.Entry, Input: input, Limits: req.Limits})
		if err != nil {
			return nil, err
		}
		out := &Result{Result: *res, Node: target}
		if len(res.Output) > inlineMax {
			if out.OutputBlob, err = s.cfg.Blobs.Store().PutBytes(res.Output); err != nil {
				return nil, err
			}
			out.Output = nil
		}
		return out, nil
	}

	s.mu.Lock()
	s.pending[target]++
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.pending[target]--
		s.mu.Unlock()
	}()

	ctx, cancel := context.WithTimeout(ctx, req.Limits.Timeout+taskGrace)
	defer cancel()
	st, err := s.node.NewStream(ctx, target, Protocol)
	if err != nil {
		return nil, err
	}
	// Resetting the stream cancels the task on the executor.
	stop := context.AfterFunc(ctx, func() { st.Reset() })
	defer stop()
	defer st.Close()

	h := header{Version: 1, ModuleHash: hash.String(), ModuleSize: len(req.Module), Entry: req.Entry, InputSize: len(req.Input), Limits: req.Limits}
	if !req.InputBlob.IsZero() {
		h.InputBlob, h.InputSize = req.InputBlob.String(), 0
		for _, p := range req.Providers {
			h.Providers = append(h.Providers, p.String())
		}
	}
	if err := writeJSON(st, h); err != nil {
		return nil, err
	}
	var o offer
	if err := readJSON(st, &o); err != nil {
		return nil, err
	}
	switch o.Status {
	case offerBusy:
		return nil, ErrBusy
	case offerRejected:
		return nil, fmt.Errorf("%w: %s", ErrRejected, o.Reason)
	case offerSendModule:
		if err := writeFrame(st, req.Module); err != nil {
			return nil, err
		}
	case offerReady:
	default:
		return nil, fmt.Errorf("%w: unexpected offer %q", ErrExecutorInternal, o.Status)
	}
	s.notePlaced(target, hash)
	if err := writeFrame(st, req.Input); err != nil {
		return nil, err
	}

	var rh resultHeader
	if err := readJSON(st, &rh); err != nil {
		return nil, err
	}
	output, err := readFrame(st, req.Limits.MaxOutputBytes)
	if err != nil {
		return nil, err
	}
	logs, err := readFrame(st, req.Limits.MaxLogBytes)
	if err != nil {
		return nil, err
	}
	if rh.ErrorKind != "" {
		return nil, &RemoteError{Kind: rh.ErrorKind, Msg: fmt.Sprintf("%s (on %s)", rh.Error, target.ShortString())}
	}
	if rh.GasUsed < 0 || rh.GasUsed > req.Limits.MaxGas {
		return nil, fmt.Errorf("%w: reported gas %d outside budget", ErrExecutorInternal, rh.GasUsed)
	}
	res := &Result{Node: target, Result: sandbox.Result{
		ExitCode: rh.ExitCode, Output: output, Logs: logs, LogTruncated: rh.LogTruncated,
		GasUsed: rh.GasUsed, Elapsed: time.Duration(rh.ElapsedNs),
	}}
	if rh.OutputBlob != "" {
		// Fetch it now, so the result outlives the executor.
		id, err := blob.ParseID(rh.OutputBlob)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrExecutorInternal, err)
		}
		if err := s.cfg.Blobs.Fetch(ctx, id, []peer.ID{target}, int64(req.Limits.MaxOutputBytes)); err != nil {
			return nil, err
		}
		res.OutputBlob = id
	}
	return res, nil
}

// notePlaced remembers that an executor holds a module, for locality.
func (s *Service) notePlaced(p peer.ID, h Hash) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.placed[p]
	if m == nil || len(m) > 1024 {
		m = map[Hash]bool{}
		s.placed[p] = m
	}
	m[h] = true
}

// outcome is what replicas must agree on: exit code, gas, output, or the
// kind of error. Logs and timings are excluded.
type outcome struct {
	exit   uint32
	gas    int64
	output blob.ID
	kind   string
}

func outcomeOf(res *Result, err error) outcome {
	if err != nil {
		return outcome{kind: ErrorKind(err)}
	}
	// Blob IDs identify content whether it arrived inline or as a blob.
	id := res.OutputBlob
	if id.IsZero() {
		id = blob.IDOf(res.Output)
	}
	return outcome{exit: res.ExitCode, gas: res.GasUsed, output: id}
}

func (s *Service) runReplicated(ctx context.Context, req Request) (*Result, error) {
	pl := newPlacement()
	type replica struct {
		res *Result
		err error
	}
	results := make([]replica, req.Replicas)
	var wg sync.WaitGroup
	for i := range req.Replicas {
		wg.Go(func() {
			res, err := s.runOnce(ctx, req, pl)
			results[i] = replica{res, err}
		})
	}
	wg.Wait()

	// Tally outcomes of replicas that ran to a verdict. Not finding an
	// executor, or executors failing, is no verdict on the task.
	counts := map[outcome]int{}
	ran := 0
	for _, r := range results {
		if r.err != nil && (errors.Is(r.err, ErrNoCapacity) || retryable(r.err) || ctx.Err() != nil) {
			continue
		}
		counts[outcomeOf(r.res, r.err)]++
		ran++
	}
	var winner outcome
	best := 0
	for o, c := range counts {
		if c > best {
			winner, best = o, c
		}
	}
	var accepted *Result
	var acceptedErr error
	var reps []Replica
	for _, r := range results {
		agreed := outcomeOf(r.res, r.err) == winner
		node := peer.ID("")
		if r.res != nil {
			node = r.res.Node
		}
		reps = append(reps, Replica{Node: node, Agreed: agreed, Err: r.err})
		if agreed && accepted == nil && acceptedErr == nil {
			accepted, acceptedErr = r.res, r.err
		}
	}
	// A majority of the requested replicas is required, whether or not the
	// rest ran.
	if best*2 <= req.Replicas {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if ran < req.Replicas {
			return nil, fmt.Errorf("%w: %d of %d replicas found an executor", ErrNotEnoughNodes, ran, req.Replicas)
		}
		return nil, fmt.Errorf("%w: no majority among %d replicas", ErrNoConsensus, req.Replicas)
	}
	if best < req.Replicas {
		s.log.Warn("replica disagreed with the majority", "replicas", req.Replicas, "agreeing", best)
	}
	if acceptedErr != nil {
		return nil, acceptedErr // the task itself fails, consistently
	}
	out := *accepted
	out.Replicas = reps
	return &out, nil
}

// MapRequest runs one module over many inputs.
type MapRequest struct {
	Module []byte
	Entry  string
	// Inputs, or InputBlobs naming stored blobs.
	Inputs     [][]byte
	InputBlobs []blob.ID
	Limits     sandbox.Limits
	// Concurrency bounds tasks in flight. Default: the free slots of all
	// known executors.
	Concurrency int
	// StopWhen, if set, ends the map early once it returns true for an item;
	// tasks still running are cancelled and not reported.
	StopWhen func(MapItem) bool
}

type MapItem struct {
	Index  int
	Result *Result
	Err    error
}

// Map runs a module over every input across the network and streams the
// results as they complete. The channel closes when all are done, the map
// is stopped, or ctx ends.
func (s *Service) Map(ctx context.Context, mr MapRequest) <-chan MapItem {
	if mr.Limits == (sandbox.Limits{}) {
		mr.Limits = sandbox.DefaultLimits()
	}
	conc := mr.Concurrency
	if conc <= 0 {
		conc = s.cfg.Slots
		for _, l := range s.Loads() {
			conc += l.Slots
		}
	}
	n := len(mr.Inputs)
	if n == 0 {
		n = len(mr.InputBlobs)
	}
	conc = max(min(conc, n), 1)

	ctx, cancel := context.WithCancel(ctx)
	out := make(chan MapItem)
	jobs := make(chan int)
	var stopped sync.Once
	var wg sync.WaitGroup
	for range conc {
		wg.Go(func() {
			for i := range jobs {
				req := Request{Module: mr.Module, Entry: mr.Entry, Limits: mr.Limits}
				if len(mr.Inputs) > 0 {
					req.Input = mr.Inputs[i]
				} else {
					req.InputBlob = mr.InputBlobs[i]
				}
				var res *Result
				prepared, unpin, err := s.prepareInput(req)
				if err == nil {
					res, err = s.runOnce(ctx, prepared, newPlacement())
					unpin()
				}
				if ctx.Err() != nil {
					continue // stopped or cancelled: not reported
				}
				item := MapItem{Index: i, Result: res, Err: err}
				select {
				case out <- item:
				case <-ctx.Done():
					continue
				}
				if mr.StopWhen != nil && mr.StopWhen(item) {
					stopped.Do(cancel)
				}
			}
		})
	}
	go func() {
	feed:
		for i := range n {
			select {
			case jobs <- i:
			case <-ctx.Done():
				break feed
			}
		}
		close(jobs)
		wg.Wait()
		cancel()
		close(out)
	}()
	return out
}
