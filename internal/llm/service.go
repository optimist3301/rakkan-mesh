package llm

import (
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"sync"
	"time"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/gpu"
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
	Serves bool    `json:"serves"`
	Shards []Shard `json:"shards"`
	// FreeMem is the memory left for weights and caches: on the GPU, if
	// Device is set. FreeCache, if set, is memory for caches apart.
	FreeMem   int64 `json:"free_mem"`
	FreeCache int64 `json:"free_cache,omitempty"`
	// Speed is how fast the node streams weights through a product, in
	// GB/s: what a forward pass costs it, per byte of layers.
	Speed    float64 `json:"speed"`
	Device   string  `json:"device,omitempty"`
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
	// Device, if set, holds the weights of stages this node serves and the
	// classifier of generations it coordinates; DeviceBytes bounds the
	// weights on it (default 85% of its memory), and MemoryBytes then
	// bounds caches only. The GPU computes the same bits as the CPU.
	Device      *gpu.Device
	DeviceBytes int64
	// Speed overrides the measured speed (GB/s of weights), for tests.
	Speed float64
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

	speed float64 // GB/s of weights, measured at start

	mu       sync.Mutex
	shards   map[Shard]*loaded
	used     int64 // bytes of caches, and of weights unless on the device
	devUsed  int64 // bytes of weights on the device
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
	onDevice bool // bytes are device memory
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
	if cfg.Device != nil && cfg.DeviceBytes <= 0 {
		cfg.DeviceBytes = cfg.Device.Info().Memory / 100 * 85
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
	s.speed = cfg.Speed
	if s.speed <= 0 && cfg.Serve {
		s.speed = MeasureSpeed(cfg.Device)
		s.log.Info("measured stage speed", "gb_per_s", fmt.Sprintf("%.1f", s.speed), "device", s.deviceName())
	}
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

// Close stops the service and frees device memory held.
func (s *Service) Close() {
	s.cancel()
	s.wg.Wait()
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, l := range s.shards {
		if l.users == 0 && l.layers != nil {
			s.drop(k)
		}
	}
	for _, cm := range s.coords {
		if cm.head.Wcls.dev != nil {
			cm.head.Wcls.dev.Free()
		}
	}
}

// NodeInfo describes a node serving stages.
type NodeInfo struct {
	ID peer.ID `json:"id"`
	// Device is the GPU holding its weights; empty for the CPU.
	Device string `json:"device,omitempty"`
	// Speed is in GB of weights per second.
	Speed float64 `json:"speed"`
	// FreeMem is the memory left for weights, FreeCache for caches when
	// they are apart (a GPU node).
	FreeMem   int64   `json:"free_mem"`
	FreeCache int64   `json:"free_cache,omitempty"`
	Shards    []Shard `json:"shards,omitempty"`
}

// Nodes lists the nodes serving stages, this one included, fastest first.
func (s *Service) Nodes() []NodeInfo {
	var out []NodeInfo
	add := func(p peer.ID, a advert) {
		out = append(out, NodeInfo{ID: p, Device: a.Device, Speed: a.Speed, FreeMem: a.FreeMem, FreeCache: a.FreeCache, Shards: a.Shards})
	}
	for p, a := range s.servers() {
		add(p, a)
	}
	if s.cfg.Serve {
		add(s.node.ID(), s.advert())
	}
	slices.SortFunc(out, func(a, b NodeInfo) int { return cmp.Compare(b.Speed, a.Speed) })
	return out
}

func (s *Service) deviceName() string {
	if s.cfg.Device == nil {
		return "cpu"
	}
	return s.cfg.Device.Info().Name
}

func (s *Service) advert() advert {
	s.mu.Lock()
	defer s.mu.Unlock()
	a := advert{Serves: s.cfg.Serve, FreeMem: s.cfg.MemoryBytes - s.used, Speed: s.speed}
	if s.cfg.Device != nil {
		// Weights and caches share the GPU's memory.
		a.Device = s.deviceName()
		a.FreeMem = s.cfg.DeviceBytes - s.devUsed
	}
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
		// Loaded outside the lock, published under it: adverts read the
		// shard meanwhile.
		manifest, layers, rope, size, err := s.load(ctx, model, from, to, providers)
		onDevice := false
		if err == nil {
			var devBytes int64
			if devBytes, onDevice = s.toDevice(manifest.Config, layers); onDevice {
				size = devBytes
			}
		}
		s.mu.Lock()
		l.manifest, l.layers, l.rope, l.bytes, l.err, l.onDevice = manifest, layers, rope, size, err, onDevice
		if l.err != nil {
			delete(s.shards, key)
		} else if !onDevice {
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
	// The host copy is needed while loading, whatever holds it after.
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

// toDevice moves loaded layers to the GPU within its budget. It reports
// the device memory taken; on failure the layers stay on the host.
func (s *Service) toDevice(c Config, layers []*Layer) (int64, bool) {
	d := s.cfg.Device
	if d == nil {
		return 0, false
	}
	need := LayerBytes(c) * int64(len(layers))
	s.mu.Lock()
	if s.devUsed+need > s.cfg.DeviceBytes {
		s.evictDevice(need)
	}
	ok := s.devUsed+need <= s.cfg.DeviceBytes
	if ok {
		s.devUsed += need
	}
	s.mu.Unlock()
	if !ok {
		s.log.Warn("GPU memory full; serving layers from the CPU")
		return 0, false
	}
	var moved int
	var err error
	for _, l := range layers {
		if _, err = l.toDevice(c, d); err != nil {
			break
		}
		moved++
	}
	if err != nil {
		for _, l := range layers[:moved] {
			l.release(c)
		}
		s.unreserveDevice(need)
		s.log.Warn("could not move layers to the GPU; serving them from the CPU", "err", err)
		return 0, false
	}
	return need, true
}

// reserveCache accounts for a stage's caches: in GPU memory, with its
// activations, when the shard's layers are there; else in RAM.
func (s *Service) reserveCache(l *loaded, positions int) (func(), error) {
	c := l.manifest.Config
	layers := len(l.layers)
	if !l.onDevice {
		n := CacheBytes(c, layers, positions)
		if err := s.reserve(n); err != nil {
			return nil, err
		}
		return func() { s.unreserve(n) }, nil
	}
	n := ArenaBytes(c, layers, positions)
	s.mu.Lock()
	if s.devUsed+n > s.cfg.DeviceBytes {
		s.evictDevice(n)
	}
	if s.devUsed+n > s.cfg.DeviceBytes {
		s.mu.Unlock()
		return nil, fmt.Errorf("GPU memory budget exhausted (%d of %d bytes in use)", s.devUsed, s.cfg.DeviceBytes)
	}
	s.devUsed += n
	s.mu.Unlock()
	return func() { s.unreserveDevice(n) }, nil
}

func (s *Service) unreserveDevice(n int64) {
	s.mu.Lock()
	s.devUsed -= n
	s.mu.Unlock()
}

// evictDevice unloads idle shards on the device until n bytes fit. The
// caller holds s.mu.
func (s *Service) evictDevice(n int64) {
	for s.devUsed+n > s.cfg.DeviceBytes {
		victim, ok := s.idlest(true)
		if !ok {
			return
		}
		s.drop(victim)
	}
}

// idlest finds the least recently used idle shard, on the device or not.
func (s *Service) idlest(onDevice bool) (Shard, bool) {
	var victim Shard
	var oldest time.Time
	for k, l := range s.shards {
		if l.users == 0 && l.layers != nil && l.onDevice == onDevice && (oldest.IsZero() || l.lastUsed.Before(oldest)) {
			victim, oldest = k, l.lastUsed
		}
	}
	return victim, !oldest.IsZero()
}

// drop unloads a shard. The caller holds s.mu.
func (s *Service) drop(k Shard) {
	l := s.shards[k]
	if l.onDevice {
		for _, layer := range l.layers {
			layer.release(l.manifest.Config)
		}
		s.devUsed -= l.bytes
	} else {
		s.used -= l.bytes
	}
	delete(s.shards, k)
}

// reserve checks the memory budget, unloading idle shards to make room.
func (s *Service) reserve(n int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.used+n > s.cfg.MemoryBytes {
		victim, ok := s.idlest(false)
		if !ok {
			return fmt.Errorf("model memory budget exhausted (%d of %d bytes in use)", s.used, s.cfg.MemoryBytes)
		}
		s.drop(victim)
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
	unreserve, err := s.reserveCache(l, positions)
	if err != nil {
		writeStatus(st, err)
		return
	}
	defer func() { unreserve(); s.notify() }()
	stage := NewStage(cfg, l.layers, l.rope, positions)
	defer stage.Close()
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
		if err = stage.ForwardMany(xs, req.Pos, req.Tokens); err != nil {
			settle(0)
			writeStatus(st, err)
			return
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
