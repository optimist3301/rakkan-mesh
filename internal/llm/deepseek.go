package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
)

// DeepSeek-V4.1 (package dsv41) in the pipeline. Its blocks are stored a
// blob each, as safetensors with the checkpoint's own tensors, but for the
// Engram tables (tens of GiB each): those are blobs of their own, which a
// stage reads a few rows of per token from its blob store instead of
// holding them in memory. What passes between stages is dsv41's Payload,
// and a coordinator embeds, hashes n-grams and classifies itself.

func parseManifestV2(m *Manifest) (*Manifest, error) {
	if m.Arch != "dsv41" {
		return nil, fmt.Errorf("model manifest: architecture %q", m.Arch)
	}
	c, err := dsv41.ParseConfig(m.ModelConfig)
	if err != nil {
		return nil, fmt.Errorf("model manifest: %w", err)
	}
	if m.Hub != nil {
		if len(m.Layers) > 0 || len(m.Engram) > 0 || m.Head != "" || m.Tokenizer != "" {
			return nil, errors.New("model manifest: parts both from the Hub and as blobs")
		}
		if err := m.Hub.check(c); err != nil {
			return nil, fmt.Errorf("model manifest: hub: %w", err)
		}
		m.ds = c
		return m, nil
	}
	if len(m.Layers) != c.Layers || len(m.Engram) != c.Layers {
		return nil, fmt.Errorf("model manifest lists %d layers and %d Engram tables, config says %d", len(m.Layers), len(m.Engram), c.Layers)
	}
	ids := append([]string{m.Head, m.Tokenizer}, m.Layers...)
	for _, id := range []string{m.Draft, m.Vision} {
		if id != "" {
			ids = append(ids, id)
		}
	}
	for i, e := range m.Engram {
		if (e != "") != slices.Contains(c.EngramLayers, i) {
			return nil, fmt.Errorf("model manifest: Engram table of layer %d", i)
		}
		if e != "" {
			ids = append(ids, e)
		}
	}
	for _, id := range ids {
		if _, err := blob.ParseID(id); err != nil {
			return nil, fmt.Errorf("model manifest: %w", err)
		}
	}
	m.ds = c
	return m, nil
}

// shape returns what planning needs of the model's layers.
func (m *Manifest) shape() shape {
	if m.ds != nil {
		return dsShape{m.ds, m.Hub}
	}
	return llamaShape{m.Config}
}

func (m *Manifest) layers() int { return m.shape().numLayers() }

// seqLen is the longest generation the model runs.
func (m *Manifest) seqLen() int {
	if m.ds != nil {
		return m.ds.MaxPositions
	}
	return m.Config.SeqLen
}

type llamaShape struct{ c Config }

func (s llamaShape) numLayers() int       { return s.c.Layers }
func (s llamaShape) layerBytes(int) int64 { return LayerBytes(s.c) }
func (s llamaShape) diskBytes(int) int64  { return 0 }
func (s llamaShape) readBytes(int) int64  { return LayerBytes(s.c) }
func (s llamaShape) hostBytes(int) int64  { return 0 }
func (s llamaShape) cacheBytes(from, to, positions int) int64 {
	return CacheBytes(s.c, to-from, positions)
}
func (s llamaShape) gas(from, to, tokens int) int64 { return StageGas(s.c, to-from, tokens) }
func (s llamaShape) payloadBytes(tokens int) int    { return 4 * tokens * s.c.Dim }
func (s llamaShape) sizes(b []byte, n int) ([]int, error) {
	if len(b) != 4*n*s.c.Dim {
		return nil, errors.New("activations of the wrong size")
	}
	out := make([]int, n)
	for i := range out {
		out[i] = 4 * s.c.Dim
	}
	return out, nil
}

type dsShape struct {
	c   *dsv41.Config
	hub *HubOrigin // where the layers come from, if from the Hub
}

func (s dsShape) numLayers() int         { return s.c.Layers }
func (s dsShape) layerBytes(i int) int64 { return dsv41.LayerBytes(s.c, i) }

// diskBytes: the Engram table kept apart, or a Hub model's files, which
// stay in the blob store as they are.
func (s dsShape) diskBytes(i int) int64 {
	if s.hub != nil {
		return hubDiskBytes(s.hub, i)
	}
	return dsv41.EngramTableBytes(s.c, i)
}
func (s dsShape) readBytes(i int) int64 { return dsv41.ActiveBytes(s.c, i) }
func (s dsShape) hostBytes(i int) int64 { return dsv41.ExpertBytes(s.c, i) }
func (s dsShape) cacheBytes(from, to, positions int) int64 {
	return dsv41.CacheBytes(s.c, from, to, positions)
}

// gas is eight per weight a token is multiplied by, as for llama models:
// the experts a token is routed to, not all of them.
func (s dsShape) gas(from, to, tokens int) int64 {
	var n int64
	for i := from; i < to; i++ {
		n += dsv41.ActiveParams(s.c, i)
	}
	return 8 * n * int64(tokens)
}
func (s dsShape) payloadBytes(tokens int) int { return tokens * dsv41.MaxPayloadBytes(s.c) }
func (s dsShape) sizes(b []byte, n int) ([]int, error) {
	out := make([]int, n)
	for i := range out {
		_, rest, err := dsv41.DecodePayload(b, s.c, 1<<30)
		if err != nil {
			return nil, err
		}
		out[i] = len(b) - len(rest)
		b = rest
	}
	if len(b) != 0 {
		return nil, errors.New("payloads of the wrong size")
	}
	return out, nil
}

// IsDeepSeek reports whether a config.json is DeepSeek-V4.1's, which
// ImportDeepSeek imports rather than ImportHF.
func IsDeepSeek(config []byte) bool {
	var c struct {
		Text struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	return json.Unmarshal(config, &c) == nil && c.Text.ModelType == "deepseek_v41_text"
}

// ImportDeepSeek stores a DeepSeek-V4.1 checkpoint (config.json,
// tokenizer.json and the safetensors shards) as blobs and returns the
// model's ID. Tensors are copied from the shards a block at a time, so
// memory stays small whatever the checkpoint's size.
func ImportDeepSeek(store *blob.Store, src HFSource, name string) (blob.ID, *Manifest, error) {
	c, err := dsv41.ParseConfig(src.Config)
	if err != nil {
		return blob.ID{}, nil, err
	}
	if _, err := ReadBPE(src.Tokenizer, c.Vocab); err != nil {
		return blob.ID{}, nil, err
	}
	tensors := map[string]*stTensor{}
	for _, f := range src.Weights {
		if err := readSafetensors(f, tensors); err != nil {
			return blob.ID{}, nil, err
		}
	}
	groups := map[int][]string{}
	var head, draft, vision []string
	for name := range tensors {
		if l, ok := dsv41.TensorLayer(name); ok && l < c.Layers {
			groups[l] = append(groups[l], name)
		} else if dsv41.IsHeadTensor(name) {
			head = append(head, name)
		} else if dsv41.IsDraftTensor(name) {
			draft = append(draft, name)
		} else if dsv41.IsVisionTensor(name) {
			vision = append(vision, name)
		}
	}
	m := &Manifest{Version: 2, Name: name, Arch: "dsv41", ModelConfig: src.Config, TokenizerFormat: "hf"}
	put := func(r io.Reader) (string, error) {
		id, _, err := store.Put(r)
		return id.String(), err
	}
	if m.Head, err = put(safetensorsReader(tensors, head)); err != nil {
		return blob.ID{}, nil, err
	}
	for i := range c.Layers {
		names := groups[i]
		engram := ""
		if slices.Contains(c.EngramLayers, i) {
			p := fmt.Sprintf("layers.%d.engram.embed.", i)
			rows, scales := tensors[p+"weight"], tensors[p+"scale"]
			if rows == nil || scales == nil {
				return blob.ID{}, nil, fmt.Errorf("layer %d: Engram table missing", i)
			}
			if engram, err = put(io.MultiReader(rows.reader(), scales.reader())); err != nil {
				return blob.ID{}, nil, err
			}
			names = slices.DeleteFunc(names, func(n string) bool { return n == p+"weight" || n == p+"scale" })
		}
		id, err := put(safetensorsReader(tensors, names))
		if err != nil {
			return blob.ID{}, nil, err
		}
		m.Layers = append(m.Layers, id)
		m.Engram = append(m.Engram, engram)
	}
	if m.Tokenizer, err = put(bytesReader(src.Tokenizer)); err != nil {
		return blob.ID{}, nil, err
	}
	if len(draft) > 0 {
		if m.Draft, err = put(safetensorsReader(tensors, draft)); err != nil {
			return blob.ID{}, nil, err
		}
	}
	if len(vision) > 0 && c.Vision.Layers > 0 {
		if m.Vision, err = put(safetensorsReader(tensors, vision)); err != nil {
			return blob.ID{}, nil, err
		}
	}
	data, _ := json.Marshal(m)
	parsed, err := ParseManifest(data)
	if err != nil {
		return blob.ID{}, nil, err
	}
	// Each part must load: shapes and types as the config says.
	if err := checkDeepSeekParts(store, parsed); err != nil {
		return blob.ID{}, nil, err
	}
	id, err := store.PutBytes(data)
	return id, parsed, err
}

// checkDeepSeekParts loads each block of an imported model once.
func checkDeepSeekParts(store *blob.Store, m *Manifest) error {
	c := m.ds
	for i := range c.Layers {
		_, closer, err := loadDeepSeekPart(store.ReadAll, store.Open, m, i, i+1)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		closer()
	}
	tok, err := store.ReadAll(mustID(m.Tokenizer), 64<<20)
	if err != nil {
		return err
	}
	_, err = newDeepseekCoord(m, tok, func() (map[string]*dsv41.Tensor, error) {
		data, err := store.ReadAll(mustID(m.Head), dsv41.HeadBytes(m.ds))
		if err != nil {
			return nil, err
		}
		return dsv41.ParseSafetensors(data)
	})
	return err
}

func mustID(s string) blob.ID {
	id, _ := blob.ParseID(s)
	return id
}

// safetensorsReader streams a safetensors file of the named tensors,
// sorted by name, from the shards they are in.
func safetensorsReader(tensors map[string]*stTensor, names []string) io.Reader {
	names = slices.Clone(names)
	sort.Strings(names)
	type entry struct {
		DType   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	h := map[string]entry{}
	var off int64
	var parts []io.Reader
	for _, n := range names {
		t := tensors[n]
		size := t.Offsets[1] - t.Offsets[0]
		h[n] = entry{t.DType, t.Shape, [2]int64{off, off + size}}
		off += size
		parts = append(parts, t.reader())
	}
	hdr, _ := json.Marshal(h)
	// The header is padded with spaces to 8 bytes, as the format asks.
	for len(hdr)%8 != 0 {
		hdr = append(hdr, ' ')
	}
	var n [8]byte
	for i := range 8 {
		n[i] = byte(uint64(len(hdr)) >> (8 * i))
	}
	return io.MultiReader(append([]io.Reader{bytesReader(n[:]), bytesReader(hdr)}, parts...)...)
}

func (t *stTensor) reader() io.Reader {
	return io.NewSectionReader(t.file, t.base+t.Offsets[0], t.Offsets[1]-t.Offsets[0])
}

type byteSlice []byte

func bytesReader(b []byte) io.Reader { return io.NewSectionReader(byteSlice(b), 0, int64(len(b))) }

func (b byteSlice) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(b)) {
		return 0, io.EOF
	}
	n := copy(p, b[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// loadDeepSeekPart builds blocks [from, to) of a model from its blobs:
// read reads a block's blob whole, open opens an Engram table's for reads
// as the blocks run. close releases the tables.
func loadDeepSeekPart(read func(blob.ID, int64) ([]byte, error), open func(blob.ID) (*blob.Reader, error),
	m *Manifest, from, to int) (*dsv41.Model, func(), error) {
	c := m.ds
	tensors := map[string]*dsv41.Tensor{}
	engram := map[int]dsv41.EngramTable{}
	var readers []*blob.Reader
	closer := func() {
		for _, r := range readers {
			r.Close()
		}
	}
	fail := func(err error) (*dsv41.Model, func(), error) {
		closer()
		return nil, nil, err
	}
	for i := from; i < to; i++ {
		data, err := read(mustID(m.Layers[i]), dsv41.LayerBytes(c, i))
		if err != nil {
			return fail(err)
		}
		ts, err := dsv41.ParseSafetensors(data)
		if err != nil {
			return fail(fmt.Errorf("layer %d: %w", i, err))
		}
		for name, t := range ts {
			if l, ok := dsv41.TensorLayer(name); !ok || l != i {
				return fail(fmt.Errorf("layer %d: tensor %s", i, name))
			}
			tensors[name] = t
		}
		if m.Engram[i] != "" {
			r, err := open(mustID(m.Engram[i]))
			if err != nil {
				return fail(err)
			}
			readers = append(readers, r)
			size := dsv41.EngramTableBytes(c, i)
			if r.Size() != size {
				return fail(fmt.Errorf("layer %d: Engram table of %d bytes, want %d", i, r.Size(), size))
			}
			at := &lockedReaderAt{r: r}
			rows := size / 33 * 32
			engram[i] = dsv41.EngramTable{Rows: io.NewSectionReader(at, 0, rows), Scales: io.NewSectionReader(at, rows, size-rows)}
		}
	}
	part, err := dsv41.NewPart(c, tensors, from, to, engram)
	if err != nil {
		return fail(err)
	}
	return part, closer, nil
}

// lockedReaderAt serializes reads of a blob: its reader caches a chunk.
type lockedReaderAt struct {
	mu sync.Mutex
	r  *blob.Reader
}

func (l *lockedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.r.ReadAt(p, off)
}

// loadDeepSeek loads a shard of a DeepSeek model on a serving node.
func (s *Service) loadDeepSeek(ctx context.Context, m *Manifest, from, to int, providers []peer.ID) (*dsv41.Model, func(), int64, error) {
	sh := m.shape()
	var size int64
	for i := from; i < to; i++ {
		size += sh.layerBytes(i)
	}
	if err := s.reserve(size); err != nil {
		return nil, nil, 0, err
	}
	defer s.unreserve(size) // accounted by the caller once loaded
	if m.Hub != nil {
		part, closer, err := s.loadHubPart(ctx, m, from, to)
		if err != nil {
			return nil, nil, 0, err
		}
		return part, closer, size, nil
	}
	for i := from; i < to; i++ {
		if err := s.blobs.Fetch(ctx, mustID(m.Layers[i]), providers, sh.layerBytes(i)); err != nil {
			return nil, nil, 0, err
		}
		if m.Engram[i] != "" {
			if err := s.blobs.Fetch(ctx, mustID(m.Engram[i]), providers, sh.diskBytes(i)); err != nil {
				return nil, nil, 0, err
			}
		}
	}
	store := s.blobs.Store()
	part, closer, err := loadDeepSeekPart(store.ReadAll, store.Open, m, from, to)
	if err != nil {
		return nil, nil, 0, err
	}
	return part, closer, size, nil
}

// deepseekStage runs a DeepSeek stage on Payloads, on the CPU or a GPU.
type deepseekStage struct {
	c     *dsv41.Config
	stage interface {
		Forward(p *dsv41.Payload, pos int) error
	}
	positions int
	done      func() // releases a GPU stage's arena
}

func (d *deepseekStage) forward(in []byte, pos, tokens int) ([]byte, error) {
	ps, err := dsv41.DecodePayloads(in, d.c, tokens, d.positions)
	if err != nil {
		return nil, err
	}
	if bs, ok := d.stage.(interface {
		ForwardBatch(ps []*dsv41.Payload, pos int) error
	}); ok {
		// A GPU stage: the weights read once for a batch of positions.
		for t := 0; t < len(ps); t += dsv41.GPUBatch {
			if err := bs.ForwardBatch(ps[t:min(t+dsv41.GPUBatch, len(ps))], pos+t); err != nil {
				return nil, err
			}
		}
		return dsv41.EncodePayloads(d.c, ps)
	}
	for t, p := range ps {
		if err := d.stage.Forward(p, pos+t); err != nil {
			return nil, err
		}
	}
	return dsv41.EncodePayloads(d.c, ps)
}

func (d *deepseekStage) close() {
	if d.done != nil {
		d.done()
	}
}

// deepseekToDevice moves a loaded DeepSeek shard's weights to the GPU
// within its budget, letting the host copy go: all of them if they fit,
// else all but the routed experts, which stay in RAM and run on the CPU
// (hostBytes). It reports the memory taken; on failure the shard stays on
// the CPU.
func (s *Service) deepseekToDevice(m *Manifest, part *dsv41.Model, from, to int) (gp *dsv41.GPUPart, devBytes, hostBytes int64, ok bool) {
	d := s.cfg.Device
	if d == nil {
		return nil, 0, 0, false
	}
	// The estimates bound what the upload takes; the actual size replaces it.
	var whole, experts int64
	for i := from; i < to; i++ {
		whole += dsv41.LayerBytes(m.ds, i)
		experts += dsv41.ExpertBytes(m.ds, i)
	}
	fits := func(n int64) bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.devUsed+n > s.cfg.DeviceBytes {
			s.evictDevice(n)
		}
		if s.devUsed+n > s.cfg.DeviceBytes {
			return false
		}
		s.devUsed += n
		return true
	}
	need, o := whole, dsv41.GPUOptions{}
	if !fits(need) {
		need, o = whole-experts, dsv41.GPUOptions{HostExperts: true}
		if !fits(need) {
			s.log.Warn("GPU memory full; serving DeepSeek layers from the CPU")
			return nil, 0, 0, false
		}
		if err := s.reserve(experts); err != nil {
			s.unreserveDevice(need)
			s.log.Warn("no RAM for the experts beside the GPU; serving DeepSeek layers from the CPU", "err", err)
			return nil, 0, 0, false
		}
		hostBytes = experts
	}
	gp, err := part.ToGPU(d, o)
	if err != nil {
		s.unreserveDevice(need)
		s.unreserve(hostBytes)
		s.log.Warn("could not move DeepSeek layers to the GPU; serving them from the CPU", "err", err)
		return nil, 0, 0, false
	}
	devBytes = gp.Bytes()
	s.unreserveDevice(need - devBytes)
	gp.DropHostWeights()
	if o.HostExperts {
		s.log.Info("DeepSeek layers on the GPU, their experts on the CPU", "from", from, "to", to)
	}
	return gp, devBytes, hostBytes, true
}

// deepseekCoord is a coordinator's part of a DeepSeek model.
type deepseekCoord struct {
	head *dsv41.Head
	tok  *BPE

	draftOnce sync.Once
	draft     *dsv41.Draft // loaded when first asked for
	draftErr  error

	visionOnce sync.Once
	vision     *dsv41.Vision // loaded when first asked for
	visionErr  error
}

// deepseekVision loads the model's image encoder, once.
func (s *Service) deepseekVision(ctx context.Context, m *Manifest, cm *deepseekCoord, providers []peer.ID) (*dsv41.Vision, error) {
	cm.visionOnce.Do(func() {
		cm.vision, cm.visionErr = s.loadVision(ctx, m, providers)
	})
	return cm.vision, cm.visionErr
}

func (s *Service) loadVision(ctx context.Context, m *Manifest, providers []peer.ID) (*dsv41.Vision, error) {
	tensors := map[string]*dsv41.Tensor{}
	switch {
	case m.ds.Vision.Layers == 0:
		return nil, errors.New("the model has no image encoder")
	case m.Hub != nil && len(m.Hub.Vision) > 0:
		all, closer, err := s.hubTensors(ctx, m.Hub, m.Hub.Vision)
		if err != nil {
			return nil, err
		}
		defer closer()
		for name, t := range all {
			if dsv41.IsVisionTensor(name) {
				if tensors[name], err = dsTensor(name, t); err != nil {
					return nil, err
				}
			}
		}
	case m.Vision != "":
		id := mustID(m.Vision)
		if err := s.blobs.Fetch(ctx, id, providers, 8<<30); err != nil {
			return nil, err
		}
		r, err := s.blobs.Store().Open(id)
		if err != nil {
			return nil, err
		}
		defer r.Close()
		all := map[string]*stTensor{}
		if err := readSafetensors(&lockedReaderAt{r: r}, all); err != nil {
			return nil, err
		}
		for name, t := range all {
			if tensors[name], err = dsTensor(name, t); err != nil {
				return nil, err
			}
		}
	default:
		return nil, errors.New("the model's manifest has no image encoder")
	}
	return dsv41.NewVision(m.ds, tensors)
}

// MaxImages bounds the images of one request.
const MaxImages = 16

// images expands each image token of tokens into its image's span: the
// span's positions are dsv41.ImagePosition in the tokens returned, and
// rows holds their inputs by position.
func (s *Service) deepseekImages(ctx context.Context, m *Manifest, cm *deepseekCoord, providers []peer.ID, tokens []int, images [][]byte) ([]int, map[int][]float32, error) {
	c := cm.head.Cfg
	n := 0
	for _, t := range tokens {
		if t == c.ImageToken {
			n++
		}
	}
	switch {
	case n != len(images):
		return nil, nil, fmt.Errorf("the prompt has %d image places, %d images were given", n, len(images))
	case n == 0:
		return tokens, nil, nil
	case len(images) > MaxImages:
		return nil, nil, fmt.Errorf("%d images, at most %d", len(images), MaxImages)
	}
	vis, err := s.deepseekVision(ctx, m, cm, providers)
	if err != nil {
		return nil, nil, fmt.Errorf("image encoder: %w", err)
	}
	out := make([]int, 0, len(tokens))
	rows := map[int][]float32{}
	k := 0
	for _, t := range tokens {
		if t != c.ImageToken {
			out = append(out, t)
			continue
		}
		img, err := c.PrepareImage(images[k])
		if err != nil {
			return nil, nil, fmt.Errorf("image %d: %w", k+1, err)
		}
		emb, err := vis.Encode(img)
		if err != nil {
			return nil, nil, err
		}
		span, err := vis.Span(img.ImageGrid, emb)
		if err != nil {
			return nil, nil, err
		}
		for _, r := range span {
			rows[len(out)] = r
			out = append(out, dsv41.ImagePosition)
		}
		k++
	}
	return out, rows, nil
}

// deepseekDraft loads the model's DSpark draft, once.
func (s *Service) deepseekDraft(ctx context.Context, m *Manifest, cm *deepseekCoord, providers []peer.ID) (*dsv41.Draft, error) {
	cm.draftOnce.Do(func() {
		cm.draft, cm.draftErr = s.loadDraft(ctx, m, providers)
	})
	return cm.draft, cm.draftErr
}

func (s *Service) loadDraft(ctx context.Context, m *Manifest, providers []peer.ID) (*dsv41.Draft, error) {
	tensors := map[string]*dsv41.Tensor{}
	switch {
	case m.Hub != nil && len(m.Hub.Draft) > 0:
		all, closer, err := s.hubTensors(ctx, m.Hub, m.Hub.Draft)
		if err != nil {
			return nil, err
		}
		defer closer()
		for name, t := range all {
			if dsv41.IsDraftTensor(name) {
				if tensors[name], err = dsTensor(name, t); err != nil {
					return nil, err
				}
			}
		}
	case m.Draft != "":
		id := mustID(m.Draft)
		r, err := func() (*blob.Reader, error) {
			if err := s.blobs.Fetch(ctx, id, providers, 64<<30); err != nil {
				return nil, err
			}
			return s.blobs.Store().Open(id)
		}()
		if err != nil {
			return nil, err
		}
		defer r.Close()
		all := map[string]*stTensor{}
		if err := readSafetensors(&lockedReaderAt{r: r}, all); err != nil {
			return nil, err
		}
		for name, t := range all {
			if tensors[name], err = dsTensor(name, t); err != nil {
				return nil, err
			}
		}
	default:
		return nil, errors.New("the model has no draft")
	}
	return dsv41.NewDraft(m.ds, tensors)
}

func (s *Service) deepseekCoordinator(ctx context.Context, m *Manifest, providers []peer.ID) (*deepseekCoord, error) {
	if o := m.Hub; o != nil {
		id, err := s.hubFile(ctx, o, "tokenizer.json")
		if err != nil {
			return nil, err
		}
		tok, err := s.blobs.Store().ReadAll(id, 64<<20)
		if err != nil {
			return nil, err
		}
		return newDeepseekCoord(m, tok, func() (map[string]*dsv41.Tensor, error) {
			all, closer, err := s.hubTensors(ctx, o, o.Head)
			if err != nil {
				return nil, err
			}
			defer closer()
			out := map[string]*dsv41.Tensor{}
			for name, t := range all {
				if dsv41.IsHeadTensor(name) {
					if out[name], err = dsTensor(name, t); err != nil {
						return nil, err
					}
				}
			}
			return out, nil
		})
	}
	part := func(idStr string, max int64) ([]byte, error) {
		id := mustID(idStr)
		if err := s.blobs.Fetch(ctx, id, providers, max); err != nil {
			return nil, err
		}
		return s.blobs.Store().ReadAll(id, max)
	}
	data, err := part(m.Tokenizer, 64<<20)
	if err != nil {
		return nil, err
	}
	return newDeepseekCoord(m, data, func() (map[string]*dsv41.Tensor, error) {
		data, err := part(m.Head, dsv41.HeadBytes(m.ds))
		if err != nil {
			return nil, err
		}
		return dsv41.ParseSafetensors(data)
	})
}

// newDeepseekCoord builds a coordinator's part from the tokenizer and the
// head's blob: the Engram hash tables come from the tokenizer.
func newDeepseekCoord(m *Manifest, tokenizer []byte, readHead func() (map[string]*dsv41.Tensor, error)) (*deepseekCoord, error) {
	c := m.ds
	tok, err := ReadBPE(tokenizer, c.Vocab)
	if err != nil {
		return nil, err
	}
	var tables *dsv41.EngramTables
	if len(c.EngramLayers) > 0 {
		if tables, err = dsv41.BuildEngramTables(c, tok); err != nil {
			return nil, err
		}
	}
	ts, err := readHead()
	if err != nil {
		return nil, err
	}
	head, err := dsv41.NewHead(c, ts, tables)
	if err != nil {
		return nil, err
	}
	return &deepseekCoord{head: head, tok: tok}, nil
}

// prompt encodes a generation's prompt: in the chat format, as one user
// message answered without reasoning, if asked, the images first. Image
// places are the model's image token, from the config: a tokenizer may
// spell the placeholder otherwise. Without chat, the placeholder in the
// text marks them.
func (d *deepseekCoord) prompt(text string, chat bool, images int) ([]int, error) {
	c := d.head.Cfg
	if chat {
		msg := dsv41.NewObject()
		msg.Set("role", dsv41.NewString("user"))
		if images == 0 {
			msg.Set("content", dsv41.NewString(text))
		} else {
			var blocks []*dsv41.Value
			for range images {
				b := dsv41.NewObject()
				b.Set("type", dsv41.NewString("image"))
				b.Set("data", dsv41.NewString("-")) // the bytes travel apart
				blocks = append(blocks, b)
			}
			b := dsv41.NewObject()
			b.Set("type", dsv41.NewString("text"))
			b.Set("text", dsv41.NewString(text))
			msg.Set("content", dsv41.NewArray(append(blocks, b)...))
		}
		p, _, err := dsv41.EncodeChat([]*dsv41.Value{msg}, dsv41.ChatOptions{ThinkingMode: "chat"})
		if err != nil {
			return nil, err
		}
		text = p
	}
	var tokens []int
	if !chat {
		tokens = append(tokens, c.BOS)
	}
	for i, piece := range strings.Split(text, dsv41.ImagePlaceholder) {
		if i > 0 {
			tokens = append(tokens, c.ImageToken)
		}
		tokens = append(tokens, d.tok.Encode(piece)...)
	}
	return tokens, nil
}

// inputs are the Payloads of positions from.. of tokens; prompt is the
// prompt's length, rows the inputs of its image positions.
func (d *deepseekCoord) inputs(tokens []int, rows map[int][]float32, from, prompt int, wantMain bool) ([]byte, error) {
	var b []byte
	for pos := from; pos < len(tokens); pos++ {
		prefill := 0
		if pos < prompt {
			prefill = prompt
		}
		var p *dsv41.Payload
		var err error
		if tokens[pos] == dsv41.ImagePosition {
			p, err = d.head.EmbedImage(tokens[:pos+1], rows[pos], prefill)
		} else {
			p, err = d.head.Embed(tokens[:pos+1], prefill)
		}
		if err != nil {
			return nil, err
		}
		p.WantMain = wantMain
		if b, err = dsv41.AppendPayload(b, d.head.Cfg, p); err != nil {
			return nil, err
		}
	}
	return b, nil
}

// outputs are the last stage's Payloads for count positions: the next
// token's scores after each, and its Main for the draft.
func (d *deepseekCoord) outputs(out []byte, count, positions int) (logits, main func(j int) []float32, err error) {
	ps, err := dsv41.DecodePayloads(out, d.head.Cfg, count, positions)
	if err != nil {
		return nil, nil, err
	}
	return func(j int) []float32 { return d.head.Logits(ps[j]) }, func(j int) []float32 { return ps[j].Main }, nil
}
