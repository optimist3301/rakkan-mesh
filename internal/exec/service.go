// Package exec distributes WebAssembly tasks across a Rakkan network.
//
// Every node runs a Service. Nodes with a worker pool execute tasks for
// other members over Protocol and gossip their load; every node schedules
// its own tasks onto the least loaded member able to run them, itself
// included. Modules are content-addressed and sent to a node only when it
// does not already have them. A task that fails because of its executor
// (connection lost, worker crash, node busy) is retried elsewhere; a task
// that fails deterministically (out of gas, trap) is not, since every node
// would fail it the same way. Determinism also makes replication
// meaningful: Run with Replicas > 1 runs a task on distinct nodes and
// accepts the result a majority returned byte for byte, gas included.
package exec

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	"github.com/optimist3301/rakkan-mesh/internal/trust"
)

var (
	ErrNoCapacity       = errors.New("no member can run this task")
	ErrBusy             = errors.New("executor busy")
	ErrRejected         = errors.New("executor rejected the task")
	ErrExecutorInternal = errors.New("executor internal error")
)

// taskGrace is how long past a task's timeout an executor waits before
// abandoning it, and a requester before giving up on an executor.
const taskGrace = 10 * time.Second

// Config configures a Service.
type Config struct {
	Node *mesh.Node
	// Blobs stores and transfers large inputs and outputs.
	Blobs *blob.Service
	// Pool executes tasks; nil makes this node a requester only.
	Pool *isolate.Pool
	// Ceiling is the most a task may request from this node.
	Ceiling sandbox.Limits
	// Slots is how many tasks run at once; it should match the pool's
	// workers. MaxQueue more may wait. Defaults: GOMAXPROCS and Slots.
	Slots, MaxQueue int
	// ModuleCacheBytes bounds modules kept for future tasks. Default 256 MiB.
	ModuleCacheBytes int
	// Capabilities are the optional host APIs this node offers; nil offers
	// all the sandbox provides (sandbox.Capabilities). Tasks needing others
	// are not scheduled here and are refused if sent anyway.
	Capabilities []string
	// Trust, when set, bounds the work guests of an open network may ask of
	// this node by their credit, and has results from guests checked
	// against independent runs. Without it every peer is trusted.
	Trust *trust.Service
	// MaxAttempts bounds executors tried per task. Default 3.
	MaxAttempts int
	// LoadInterval is how often load is gossiped. Default 3s.
	LoadInterval time.Duration
	Logger       *slog.Logger
}

// Load is a node's capacity, as gossiped.
type Load struct {
	Slots   int            `json:"slots"`
	Busy    int            `json:"busy"`
	Queued  int            `json:"queued"`
	Ceiling sandbox.Limits `json:"ceiling"`
	CPUs    int            `json:"cpus"`
	Arch    string         `json:"arch"`
	// Capabilities are the optional host APIs the node offers (see
	// sandbox.Capabilities).
	Capabilities []string `json:"capabilities,omitempty"`

	received time.Time
}

// Service runs and schedules tasks.
type Service struct {
	cfg     Config
	node    *mesh.Node
	log     *slog.Logger
	modules *ModuleStore
	topic   *pubsub.Topic

	inflight atomic.Int64 // tasks admitted for local execution
	changed  chan struct{}

	// For tests: modules received, and a hook that tampers with results.
	modulesReceived atomic.Int64
	tamper          func(*sandbox.Result)

	mu      sync.Mutex
	loads   map[peer.ID]Load
	pending map[peer.ID]int           // our tasks in flight at each executor
	placed  map[peer.ID]map[Hash]bool // modules an executor is known to hold
	caps    map[Hash][]string         // capabilities each module needs

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// New starts a service on a running node.
func New(cfg Config) (*Service, error) {
	if cfg.Node == nil || cfg.Blobs == nil {
		return nil, errors.New("exec: node and blob service are required")
	}
	if cfg.Slots <= 0 {
		cfg.Slots = runtime.GOMAXPROCS(0)
	}
	if cfg.MaxQueue <= 0 {
		cfg.MaxQueue = cfg.Slots
	}
	if cfg.ModuleCacheBytes <= 0 {
		cfg.ModuleCacheBytes = 256 << 20
	}
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 3
	}
	if cfg.LoadInterval <= 0 {
		cfg.LoadInterval = 3 * time.Second
	}
	if cfg.Ceiling == (sandbox.Limits{}) {
		cfg.Ceiling = sandbox.DefaultLimits()
	}
	if cfg.Capabilities == nil {
		cfg.Capabilities = sandbox.Capabilities()
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	s := &Service{
		cfg:     cfg,
		node:    cfg.Node,
		log:     cfg.Logger,
		modules: NewModuleStore(cfg.ModuleCacheBytes),
		changed: make(chan struct{}, 1),
		loads:   map[peer.ID]Load{},
		pending: map[peer.ID]int{},
		placed:  map[peer.ID]map[Hash]bool{},
		caps:    map[Hash][]string{},
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())

	var err error
	if s.topic, err = cfg.Node.JoinTopic("load/1"); err != nil {
		return nil, fmt.Errorf("exec: %w", err)
	}
	sub, err := s.topic.Subscribe()
	if err != nil {
		return nil, fmt.Errorf("exec: %w", err)
	}
	s.wg.Go(func() { s.receiveLoads(sub) })
	s.wg.Go(s.publishLoads)
	if cfg.Pool != nil {
		cfg.Node.Handle(Protocol, s.handle)
	}
	return s, nil
}

// Close stops the service. Tasks in flight are not interrupted.
func (s *Service) Close() {
	s.cancel()
	s.wg.Wait()
}

// Modules is the local module store; uploaded modules can be run by hash.
func (s *Service) Modules() *ModuleStore { return s.modules }

// LocalLoad is this node's capacity.
func (s *Service) LocalLoad() Load {
	l := Load{Ceiling: s.cfg.Ceiling, CPUs: runtime.NumCPU(), Arch: runtime.GOARCH}
	if s.cfg.Pool == nil {
		return l
	}
	n := int(s.inflight.Load())
	l.Capabilities = s.cfg.Capabilities
	l.Slots = s.cfg.Slots
	l.Busy = min(n, s.cfg.Slots)
	l.Queued = max(n-s.cfg.Slots, 0)
	return l
}

// Loads returns the recently gossiped load of other members.
func (s *Service) Loads() map[peer.ID]Load {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[peer.ID]Load, len(s.loads))
	for p, l := range s.loads {
		if s.fresh(l) {
			out[p] = l
		}
	}
	return out
}

func (s *Service) fresh(l Load) bool { return time.Since(l.received) < 3*s.cfg.LoadInterval }

func (s *Service) notifyChanged() {
	select {
	case s.changed <- struct{}{}:
	default:
	}
}

// publishLoads gossips local load periodically and soon after it changes.
func (s *Service) publishLoads() {
	tick := time.NewTicker(s.cfg.LoadInterval)
	defer tick.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-tick.C:
		case <-s.changed:
			// Coalesce bursts of task starts and ends.
			select {
			case <-time.After(200 * time.Millisecond):
			case <-s.ctx.Done():
				return
			}
		}
		b, _ := json.Marshal(s.LocalLoad())
		s.topic.Publish(s.ctx, b)
	}
}

func (s *Service) receiveLoads(sub *pubsub.Subscription) {
	for {
		m, err := sub.Next(s.ctx)
		if err != nil {
			return
		}
		from := m.GetFrom()
		if from == s.node.ID() || !s.node.IsMember(from) {
			continue
		}
		var l Load
		if json.Unmarshal(m.Data, &l) != nil {
			continue
		}
		l.received = time.Now()
		s.mu.Lock()
		s.loads[from] = l
		s.mu.Unlock()
	}
}

// admit reserves local capacity for a task.
func (s *Service) admit() bool {
	if s.inflight.Add(1) > int64(s.cfg.Slots+s.cfg.MaxQueue) {
		s.inflight.Add(-1)
		return false
	}
	s.notifyChanged()
	return true
}

func (s *Service) release() {
	s.inflight.Add(-1)
	s.notifyChanged()
}

// handle executes a task for another member.
func (s *Service) handle(st network.Stream, c *membership.Cert) {
	defer st.Close()
	st.SetReadDeadline(time.Now().Add(30 * time.Second))
	var h header
	if err := readJSON(st, &h); err != nil {
		st.Reset()
		return
	}
	hash, ok := ParseHash(h.ModuleHash)
	for _, b := range append([]string{h.InputBlob}, h.Blobs...) {
		if _, err := blob.ParseID(b); b != "" && err != nil {
			ok = false
		}
	}
	if len(h.Blobs) > sandbox.MaxBlobs {
		ok = false
	}
	switch {
	case !ok || h.Version != 1:
		writeJSON(st, offer{Status: offerRejected, Reason: "malformed request"})
		return
	case h.Limits.Within(s.cfg.Ceiling) != nil:
		writeJSON(st, offer{Status: offerRejected, Reason: h.Limits.Within(s.cfg.Ceiling).Error()})
		return
	case h.ModuleSize > h.Limits.MaxModuleBytes || h.InputSize > h.Limits.MaxInputBytes || h.ModuleSize < 0 || h.InputSize < 0 ||
		h.InputBlob != "" && h.InputSize != 0:
		writeJSON(st, offer{Status: offerRejected, Reason: "module or input exceeds the requested limits"})
		return
	}
	settle := func(int64) {}
	if s.cfg.Trust != nil {
		var err error
		if settle, err = s.cfg.Trust.Reserve(c.Subject, h.Limits.MaxGas); err != nil {
			writeJSON(st, offer{Status: offerRejected, Reason: err.Error()})
			return
		}
	}
	if !s.admit() {
		settle(0)
		writeJSON(st, offer{Status: offerBusy})
		return
	}
	defer s.release()
	var gasUsed int64 // charged to the requester's credit
	defer func() { settle(gasUsed) }()

	module, have := s.modules.Get(hash)
	status := offerReady
	if !have {
		status = offerSendModule
	}
	if writeJSON(st, offer{Status: status}) != nil {
		return
	}
	if !have {
		var err error
		if module, err = readFrame(st, h.ModuleSize); err != nil || len(module) != h.ModuleSize || HashOf(module) != hash {
			st.Reset()
			return
		}
		s.modules.Put(module)
		s.modulesReceived.Add(1)
	}
	if caps, err := s.capabilities(hash, module); err != nil {
		writeResult(st, nil, err, s.cfg.Blobs.Store())
		return
	} else if c := missingCapability(caps, s.cfg.Capabilities); c != "" {
		// Misrouted, perhaps on stale load information: not the task's
		// fault, so the requester tries elsewhere.
		writeResult(st, nil, fmt.Errorf("%w: capability %q not offered by this node", ErrRejected, c), s.cfg.Blobs.Store())
		return
	}
	input, err := readFrame(st, h.InputSize)
	if err != nil || len(input) != h.InputSize {
		st.Reset()
		return
	}
	st.SetReadDeadline(time.Time{})

	ctx, cancel := context.WithTimeout(s.ctx, h.Limits.Timeout+taskGrace)
	defer cancel()
	providers := []peer.ID{c.Subject}
	for _, p := range h.Providers {
		if id, err := peer.Decode(p); err == nil && id != c.Subject {
			providers = append(providers, id)
		}
	}
	if h.InputBlob != "" {
		if input, err = s.readInputBlob(ctx, h.InputBlob, providers, h.Limits); err != nil {
			writeResult(st, nil, err, s.cfg.Blobs.Store())
			return
		}
	}
	var attached []blob.ID
	for _, b := range h.Blobs {
		id, _ := blob.ParseID(b) // validated with the header
		attached = append(attached, id)
	}
	sources, closeSources, err := s.openBlobs(ctx, attached, providers)
	if err != nil {
		writeResult(st, nil, err, s.cfg.Blobs.Store())
		return
	}
	defer closeSources()
	// The requester resets the stream to cancel; it reads to EOF only after
	// receiving the result.
	go func() {
		st.Read(make([]byte, 1))
		cancel()
	}()
	res, err := s.cfg.Pool.Run(ctx, c.Subject.String(), sandbox.Task{Module: module, Entry: h.Entry, Input: input, Blobs: sources, Limits: h.Limits})
	if ctx.Err() != nil {
		// Cancelled by the requester, or this node is shutting down: no
		// verdict on the task. Resetting tells the requester to go elsewhere.
		st.Reset()
		return
	}
	switch {
	case err == nil:
		gasUsed = res.GasUsed
	case errors.Is(err, sandbox.ErrOutOfGas):
		gasUsed = h.Limits.MaxGas
	}
	if err == nil && s.tamper != nil {
		s.tamper(res)
	}
	writeResult(st, res, err, s.cfg.Blobs.Store())
}

// openBlobs opens a task's attached blobs for reading on demand.
func (s *Service) openBlobs(ctx context.Context, ids []blob.ID, providers []peer.ID) ([]sandbox.BlobSource, func(), error) {
	var opened []blob.Source
	closeAll := func() {
		for _, o := range opened {
			o.Close()
		}
	}
	var sources []sandbox.BlobSource
	for _, id := range ids {
		src, err := s.cfg.Blobs.OpenRemote(ctx, id, providers)
		if err != nil {
			closeAll()
			return nil, nil, fmt.Errorf("%w: %v", sandbox.ErrBlobUnavailable, err)
		}
		opened = append(opened, src)
		sources = append(sources, src)
	}
	return sources, closeAll, nil
}

// readInputBlob fetches a task's input blob and reads it whole: the worker
// receives its input in one message.
func (s *Service) readInputBlob(ctx context.Context, idStr string, providers []peer.ID, lim sandbox.Limits) ([]byte, error) {
	id, err := blob.ParseID(idStr) // validated with the header
	if err != nil {
		return nil, err
	}
	if err := s.cfg.Blobs.Fetch(ctx, id, providers, int64(lim.MaxInputBytes)); err != nil {
		if errors.Is(err, blob.ErrTooLarge) {
			return nil, fmt.Errorf("%w: %v", sandbox.ErrInputTooLarge, err)
		}
		return nil, err
	}
	return s.cfg.Blobs.Store().ReadAll(id, int64(lim.MaxInputBytes))
}

// writeResult sends a task's outcome; an output larger than inlineMax is
// stored as a blob for the requester to fetch.
func writeResult(w io.Writer, res *sandbox.Result, err error, store *blob.Store) error {
	rh := resultHeader{}
	var output, logs []byte
	if err != nil {
		rh.ErrorKind, rh.Error = ErrorKind(err), err.Error()
	} else {
		rh.ExitCode, rh.GasUsed, rh.ElapsedNs = res.ExitCode, res.GasUsed, int64(res.Elapsed)
		rh.LogTruncated = res.LogTruncated
		output, logs = res.Output, res.Logs
		rh.OutputSize, rh.LogsSize = len(output), len(logs)
		if len(output) > inlineMax {
			id, perr := store.PutBytes(output)
			if perr != nil {
				return perr
			}
			rh.OutputBlob, output = id.String(), nil
		}
	}
	if err := writeJSON(w, rh); err != nil {
		return err
	}
	if err := writeFrame(w, output); err != nil {
		return err
	}
	return writeFrame(w, logs)
}
