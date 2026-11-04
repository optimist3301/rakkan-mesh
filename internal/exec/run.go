package exec

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/meter"
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
	// Blobs are attached read-only data the task reads in pieces with
	// blob_read; executors fetch only the chunks read. At most
	// sandbox.MaxBlobs.
	Blobs []blob.ID
	// Providers are other members holding InputBlob or Blobs, which
	// executors may fetch from in parallel with this node.
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
	// accepted outcome, when the task was replicated or its result checked.
	Replicas []Replica
	// Checked is set when an untrusted guest's result was confirmed by an
	// independent run (see Config.Trust).
	Checked bool
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
	return s.runChecked(ctx, req, newPlacement())
}

// prepareInput turns a large inline input into a blob, so retries and
// replicas share one stored copy and executors fetch it in verified chunks,
// and pins the task's blobs held here while it runs. Attached blobs may
// instead be held only by Providers.
func (s *Service) prepareInput(req Request) (Request, func(), error) {
	store := s.cfg.Blobs.Store()
	if len(req.Blobs) > sandbox.MaxBlobs {
		return req, nil, fmt.Errorf("%w: %d blobs attached, limit %d", sandbox.ErrLimitPolicy, len(req.Blobs), sandbox.MaxBlobs)
	}
	if req.InputBlob.IsZero() && len(req.Input) > inlineMax {
		id, err := store.PutBytes(req.Input)
		if err != nil {
			return req, nil, err
		}
		req.InputBlob, req.Input = id, nil
	}
	var unpins []func()
	unpinAll := func() {
		for _, u := range unpins {
			u()
		}
	}
	if !req.InputBlob.IsZero() {
		unpin, ok := store.Pin(req.InputBlob)
		if !ok {
			return req, nil, fmt.Errorf("%w: input blob %s", blob.ErrNotFound, req.InputBlob)
		}
		unpins = append(unpins, unpin)
	}
	for _, id := range req.Blobs {
		if unpin, ok := store.Pin(id); ok {
			unpins = append(unpins, unpin)
		} else if len(req.Providers) == 0 {
			unpinAll()
			return req, nil, fmt.Errorf("%w: attached blob %s, and no providers", blob.ErrNotFound, id)
		}
	}
	return req, unpinAll, nil
}

// capabilities returns the host APIs a module needs, cached by hash.
func (s *Service) capabilities(hash Hash, module []byte) ([]string, error) {
	s.mu.Lock()
	caps, ok := s.caps[hash]
	s.mu.Unlock()
	if ok {
		return caps, nil
	}
	mods, err := meter.ImportModules(module)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", sandbox.ErrInvalidModule, err)
	}
	caps = sandbox.CapabilitiesFor(mods)
	s.mu.Lock()
	if len(s.caps) > 4096 {
		clear(s.caps)
	}
	s.caps[hash] = caps
	s.mu.Unlock()
	return caps, nil
}

// missingCapability returns the first capability in need that have lacks.
func missingCapability(need, have []string) string {
	for _, c := range need {
		if !slices.Contains(have, c) {
			return c
		}
	}
	return ""
}

// placement tracks executors already used by a task, so retries and
// replicas go to different ones.
type placement struct {
	mu   sync.Mutex
	used map[peer.ID]bool
	// trusted limits executors to this node and members, while any are
	// able to run the task.
	trusted bool
}

func newPlacement() *placement { return &placement{used: map[peer.ID]bool{}} }

// distrust sends the task's remaining attempts to trusted executors, when
// any can run it, after a guest failed it.
func (pl *placement) distrust() {
	pl.mu.Lock()
	pl.trusted = true
	pl.mu.Unlock()
}

// runOnce runs a task on one executor, retrying on others when the
// executor fails. Busy executors are waited out without counting as an
// attempt; the context bounds the wait.
func (s *Service) runOnce(ctx context.Context, req Request, pl *placement) (*Result, error) {
	hash := HashOf(req.Module)
	caps, err := s.capabilities(hash, req.Module)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 1; attempt <= s.cfg.MaxAttempts; {
		target, anyCandidate, limitErr := s.pick(hash, req.Limits, caps, pl)
		if target == "" {
			if !anyCandidate {
				switch {
				case lastErr != nil:
					return nil, fmt.Errorf("%w (last attempt: %v)", ErrNoCapacity, lastErr)
				case limitErr != nil:
					return nil, fmt.Errorf("%w: %v", ErrNoCapacity, limitErr)
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
			s.forgive(target)
			res.Attempts = attempt
			return res, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		guest := target != s.node.ID() && !s.node.IsCertified(target)
		switch {
		case errors.Is(err, ErrBusy) && guest:
			// Its load said it had room. Not an attempt, but it gets no
			// more of this task, and the rest goes to trusted executors.
			s.penalize(target)
			pl.distrust()
			continue
		case errors.Is(err, ErrBusy):
			s.markFull(target)
			s.release1(pl, target) // it may have room again later
			continue
		case !retryable(err):
			return nil, err
		}
		if guest {
			s.penalize(target)
			pl.distrust()
		}
		s.log.Info("executor failed, retrying elsewhere", "peer", target.ShortString(), "err", err)
		if s.cfg.Trust != nil && !errors.Is(err, ErrRejected) {
			s.cfg.Trust.Failed(target)
		}
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
		case "worker_crashed", "worker_protocol", "internal", "canceled", "blob_unavailable":
			return true
		}
		return false
	}
	if errors.Is(err, ErrRejected) {
		return true // another node may have a higher ceiling
	}
	// Local sandbox errors are deterministic; anything else is transport.
	switch ErrorKind(err) {
	case "internal", "worker_crashed", "worker_protocol", "blob_unavailable":
		return true
	}
	return false
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
func (s *Service) pick(hash Hash, lim sandbox.Limits, caps []string, pl *placement) (target peer.ID, anyCandidate bool, limitErr error) {
	type cand struct {
		id    peer.ID
		score float64
		rtt   time.Duration
	}
	var cands []cand
	self := s.node.ID()
	pl.mu.Lock()
	defer pl.mu.Unlock()

	// refuse records why an executor cannot take the task at all.
	refuse := func(load Load) error {
		if c := missingCapability(caps, load.Capabilities); c != "" {
			return fmt.Errorf("no executor offers capability %q", c)
		}
		if err := lim.Within(load.Ceiling); err != nil {
			return fmt.Errorf("the task's limits exceed every executor's policy (%v)", err)
		}
		return nil
	}
	if s.cfg.Pool != nil && !pl.used[self] {
		limitErr = refuse(s.LocalLoad())
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
	if tr := s.cfg.Trust; tr != nil {
		// Leave out peers caught cheating, and guests when a trusted
		// executor is wanted and one is able to run the task.
		keep := members[:0:0]
		trustedAble := false
		for _, m := range members {
			if tr.Suspect(m.ID) {
				continue
			}
			keep = append(keep, m)
			if !m.Guest {
				trustedAble = trustedAble || s.able(m.ID, caps, lim)
			}
		}
		if pl.trusted && (trustedAble || s.cfg.Pool != nil && limitErr == nil && !pl.used[self]) {
			members = slices.DeleteFunc(keep, func(m mesh.Member) bool { return m.Guest })
		} else {
			members = keep
		}
	}
	s.mu.Lock()
	for _, m := range members {
		l, ok := s.loads[m.ID]
		if !m.Connected || !ok || !s.fresh(l) || l.Slots == 0 || pl.used[m.ID] || s.penalized(m.ID) {
			continue
		}
		if err := refuse(l); err != nil {
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
		if m.Guest && s.cfg.Trust != nil {
			// Prefer members, then reputable guests, at equal capacity:
			// their results need fewer checks.
			score -= 0.5 * (1 - s.cfg.Trust.Score(m.ID))
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
		sources, closeSources, err := s.openBlobs(ctx, req.Blobs, req.Providers)
		if err != nil {
			return nil, err
		}
		defer closeSources()
		input := req.Input
		if !req.InputBlob.IsZero() {
			if input, err = s.cfg.Blobs.Store().ReadAll(req.InputBlob, int64(req.Limits.MaxInputBytes)); err != nil {
				if errors.Is(err, blob.ErrTooLarge) {
					return nil, fmt.Errorf("%w: %v", sandbox.ErrInputTooLarge, err)
				}
				return nil, err
			}
		}
		res, err := s.cfg.Pool.Run(ctx, string(target), sandbox.Task{Module: req.Module, Entry: req.Entry, Input: input, Blobs: sources, Limits: req.Limits})
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
	}
	for _, id := range req.Blobs {
		h.Blobs = append(h.Blobs, id.String())
	}
	for _, p := range req.Providers {
		h.Providers = append(h.Providers, p.String())
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
		if s.cfg.Trust != nil && rh.ErrorKind == ErrorKind(sandbox.ErrOutOfGas) {
			s.cfg.Trust.Consumed(target, req.Limits.MaxGas)
		}
		return nil, &RemoteError{Kind: rh.ErrorKind, Msg: fmt.Sprintf("%s (on %s)", rh.Error, target.ShortString()), Node: target}
	}
	if rh.GasUsed < 0 || rh.GasUsed > req.Limits.MaxGas {
		return nil, fmt.Errorf("%w: reported gas %d outside budget", ErrExecutorInternal, rh.GasUsed)
	}
	if s.cfg.Trust != nil {
		s.cfg.Trust.Consumed(target, rh.GasUsed)
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
		if !verdict(ctx, r.err) {
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
		reps = append(reps, Replica{Node: executorOf(r.res, r.err), Agreed: agreed, Err: r.err})
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
	if s.cfg.Trust != nil {
		for i, r := range results {
			if !verdict(ctx, r.err) {
				continue
			}
			s.judge(executorOf(r.res, r.err), reps[i].Agreed)
		}
	}
	if acceptedErr != nil {
		return nil, acceptedErr // the task itself fails, consistently
	}
	out := *accepted
	out.Replicas = reps
	return &out, nil
}

// verdict reports whether a run's outcome is a verdict on the task: it
// succeeded or failed deterministically, rather than finding no executor or
// losing it.
func verdict(ctx context.Context, err error) bool {
	return err == nil || !(errors.Is(err, ErrNoCapacity) || retryable(err) || ctx.Err() != nil)
}

// executorOf returns the node that produced a result or a task error.
func executorOf(res *Result, err error) peer.ID {
	if res != nil {
		return res.Node
	}
	var re *RemoteError
	if errors.As(err, &re) {
		return re.Node
	}
	return ""
}

// judge records whether an executor's outcome agreed with the accepted one.
func (s *Service) judge(p peer.ID, agreed bool) {
	switch {
	case p == "":
	case agreed:
		s.cfg.Trust.Verified(p)
	default:
		s.cfg.Trust.Cheated(p, "task outcome disagreed with independent runs")
	}
}

// able reports whether a member's gossiped load says it can run a task.
func (s *Service) able(p peer.ID, caps []string, lim sandbox.Limits) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.loads[p]
	return ok && s.fresh(l) && l.Slots > 0 && missingCapability(caps, l.Capabilities) == "" && lim.Within(l.Ceiling) == nil
}

// runChecked runs a task once and, when an untrusted guest produced the
// outcome, has it checked (see trust.Service.ShouldVerify): the task runs
// again elsewhere, on a trusted executor where possible, and a third run
// breaks a tie between untrusted ones. Whoever disagrees with the accepted
// outcome is reported as cheating. Determinism makes this exact: honest
// executors agree byte for byte, gas included.
func (s *Service) runChecked(ctx context.Context, req Request, pl *placement) (*Result, error) {
	res, err := s.runOnce(ctx, req, pl)
	first := executorOf(res, err)
	tr := s.cfg.Trust
	if tr == nil || first == "" || !verdict(ctx, err) || !tr.ShouldVerify(first) {
		return res, err
	}
	type run struct {
		node peer.ID
		res  *Result
		err  error
	}
	runs := []run{{first, res, err}}
	pl.mu.Lock()
	pl.trusted = true
	pl.mu.Unlock()
	for len(runs) < 3 {
		r, e := s.runOnce(ctx, req, pl)
		if !verdict(ctx, e) {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			// Nobody else could run it: the result stands unchecked.
			s.log.Warn("could not check a guest's result", "peer", first.ShortString(), "err", e)
			return res, err
		}
		runs = append(runs, run{executorOf(r, e), r, e})
		// Agreement, or a trusted dissent, settles it.
		a, b := runs[0], runs[len(runs)-1]
		if outcomeOf(a.res, a.err) == outcomeOf(b.res, b.err) || tr.Trusted(b.node) {
			break
		}
	}
	counts := map[outcome]int{}
	for _, r := range runs {
		counts[outcomeOf(r.res, r.err)]++
	}
	last := runs[len(runs)-1]
	accepted := outcomeOf(last.res, last.err)
	if !tr.Trusted(last.node) && counts[accepted] < 2 {
		// Three untrusted runs, three outcomes.
		return nil, fmt.Errorf("%w: three executors returned different results", ErrNoConsensus)
	}
	var reps []Replica
	var out *Result
	var outErr error
	for _, r := range runs {
		agreed := outcomeOf(r.res, r.err) == accepted
		s.judge(r.node, agreed)
		reps = append(reps, Replica{Node: r.node, Agreed: agreed, Err: r.err})
		if agreed && out == nil && outErr == nil {
			out, outErr = r.res, r.err
		}
	}
	if outErr != nil {
		return nil, outErr
	}
	o := *out
	o.Replicas, o.Checked, o.Attempts = reps, true, len(runs)
	return &o, nil
}

// MapRequest runs one module over many inputs.
type MapRequest struct {
	Module []byte
	Entry  string
	// Inputs, or InputBlobs naming stored blobs.
	Inputs     [][]byte
	InputBlobs []blob.ID
	// Blobs are attached to every task: shared data such as a model or a
	// dictionary, fetched once per executor.
	Blobs     []blob.ID
	Providers []peer.ID
	Limits    sandbox.Limits
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
				req := Request{Module: mr.Module, Entry: mr.Entry, Limits: mr.Limits, Blobs: mr.Blobs, Providers: mr.Providers}
				if len(mr.Inputs) > 0 {
					req.Input = mr.Inputs[i]
				} else {
					req.InputBlob = mr.InputBlobs[i]
				}
				var res *Result
				prepared, unpin, err := s.prepareInput(req)
				if err == nil {
					res, err = s.runChecked(ctx, prepared, newPlacement())
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
