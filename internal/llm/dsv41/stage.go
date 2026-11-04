package dsv41

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// A pipeline runs a model's blocks in stages on different nodes, one
// position after another. The coordinator embeds each token and hashes its
// n-grams (Head.Embed), each stage runs its blocks on the Payload
// (Stage.Forward) and hands it to the next, and the coordinator turns the
// last stage's Payload into logits (Head.Logits). The stages compute what
// one process computes, bit for bit, whatever the split: the Payload
// carries everything a block reads that earlier blocks made.
//
// Besides the stream that is the compressed attention's shared state. Layers
// after a KV source read its caches, which grow by at most one entry per
// position: the Payload carries the new entries, and a stage keeps a copy
// of the caches of the source before its first layer. The selection of the
// latest index source and the candidate pool are made anew at every
// position, and travel whole.

// Payload is what passes between stages for one position.
type Payload struct {
	// Prefill is the prompt's length while the prompt runs, else 0.
	Prefill int
	H       []float32 // [hc][dim], bf16 values: the stream
	PreMix  []float32 // [hc]: how the next block mixes its input
	Hashes  [][]int64 // per Engram layer: the table rows of this position's n-grams
	// Source is the latest KV source run, -1 for none; CompKV and IndexK
	// the entries it cached at this position, if any.
	Source         int
	CompKV, IndexK []float32
	// TopK is the latest index source's selection, -1 for empty slots;
	// Candidates the candidate pool, nil before the candidate source.
	TopK       []int
	Candidates []bool
	// WantMain asks the stages for Main: the stream's mean (bf16) at the
	// input of each DSpark target layer passed, dim values each, in layer
	// order, for the draft model.
	WantMain bool
	Main     []float32
}

// Embed starts a position's Payload: history holds the tokens so far, the
// position's own last (only the last Engram n-gram's worth are read).
// prefill is the prompt's length while it runs, else 0.
func (h *Head) Embed(history []int, prefill int) (*Payload, error) {
	c := h.Cfg
	if len(history) == 0 {
		return nil, errors.New("dsv41: no token")
	}
	for _, t := range history[max(len(history)-max(c.EngramMaxNgram, 1), 0):] {
		if t < 0 || t >= c.Vocab {
			return nil, fmt.Errorf("dsv41: token %d outside the vocabulary of %d", t, c.Vocab)
		}
	}
	token := history[len(history)-1]
	dim, hc := c.Dim, c.HCMult
	p := &Payload{Prefill: prefill, H: make([]float32, hc*dim), PreMix: make([]float32, hc), Source: -1}
	for k := range hc {
		for d := range dim {
			p.H[k*dim+d] = num.BF16(h.embed[token*dim+d])
		}
	}
	p.PreMix[0] = 1
	if len(c.EngramLayers) > 0 {
		n := h.engram.MaxNgram
		var ids []int64
		for _, t := range history[max(len(history)-n, 0):] {
			ids = append(ids, h.engram.TokenMap[t])
		}
		p.Hashes = h.engram.hash(ids)
	}
	return p, nil
}

// Logits finishes a position: the classifier's scores of the next token.
func (h *Head) Logits(p *Payload) []float32 {
	c := h.Cfg
	x := make([]float32, c.Dim)
	hcPre(x, p.H, p.PreMix)
	rmsNorm(x, x, h.norm, c.NormEps)
	logits := make([]float32, c.Vocab)
	h.head.apply(logits, x, nil)
	return logits
}

// Stage runs a part's blocks for one generation.
type Stage struct {
	st *State
}

// NewStage starts a generation on the blocks of m.
func (m *Model) NewStage() *Stage { return &Stage{st: m.NewState()} }

// Forward runs the blocks on the Payload of position pos in place.
// Positions come in order from 0; a failed stage is replaced by a new one
// replaying them.
// A position already run is rewound to first (State.Rewind): a draft's
// verification goes on from the last accepted position.
func (s *Stage) Forward(p *Payload, pos int) error {
	if s.st.m.hostless {
		return errors.New("dsv41: the part's weights are on a GPU only")
	}
	if pos < s.st.pos {
		if err := s.st.Rewind(pos); err != nil {
			return err
		}
	}
	return s.st.run(p, pos)
}

// Pos is the next position the stage expects.
func (s *Stage) Pos() int { return s.st.pos }

// Payload encoding, little-endian:
//
//	prefill u32, source i32, flags u32 (1: CompKV, 2: IndexK, 4: Candidates)
//	H [hc·dim] bf16, PreMix [hc] f32, Hashes [engram layers][rows] i64
//	CompKV [head dim] f32, IndexK [index dim] f32, when flagged
//	TopK: count u32, [count] i32
//	Candidates, when flagged: length u32, blocks u32, [blocks] u32: the
//	pool is whole blocks of the candidate block size, listed ascending
const (
	hasCompKV = 1 << iota
	hasIndexK
	hasCandidates
	hasMain
)

// MaxPayloadBytes bounds a position's encoded Payload.
func MaxPayloadBytes(c *Config) int {
	n := 12 + 2*c.HCMult*c.Dim + 4*c.HCMult
	n += 8 * len(c.EngramLayers) * (c.EngramMaxNgram - 1) * c.EngramHeads
	n += 4*c.HeadDim + 4*c.IndexDim
	n += 4 + 4*c.IndexTopK
	n += 8 + 4*c.CandidateBlocks
	n += 4 + 2*c.Dim*len(c.DraftTargets)
	return n
}

// AppendPayload appends p's encoding to b.
func AppendPayload(b []byte, c *Config, p *Payload) ([]byte, error) {
	le := binary.LittleEndian
	flags := uint32(0)
	if p.CompKV != nil {
		flags |= hasCompKV
	}
	if p.IndexK != nil {
		flags |= hasIndexK
	}
	if p.Candidates != nil {
		flags |= hasCandidates
	}
	if p.WantMain {
		flags |= hasMain
	}
	if len(p.Main)%c.Dim != 0 || len(p.Main) > c.Dim*len(c.DraftTargets) || !p.WantMain && len(p.Main) > 0 {
		return nil, errors.New("dsv41: payload does not fit the model")
	}
	if len(p.H) != c.HCMult*c.Dim || len(p.PreMix) != c.HCMult || len(p.Hashes) != len(c.EngramLayers) ||
		p.CompKV != nil && len(p.CompKV) != c.HeadDim || p.IndexK != nil && len(p.IndexK) != c.IndexDim || len(p.TopK) > c.IndexTopK {
		return nil, errors.New("dsv41: payload does not fit the model")
	}
	b = le.AppendUint32(b, uint32(p.Prefill))
	b = le.AppendUint32(b, uint32(int32(p.Source)))
	b = le.AppendUint32(b, flags)
	for _, v := range p.H {
		bits := math.Float32bits(v)
		if bits&0xffff != 0 {
			return nil, errors.New("dsv41: stream value not bf16")
		}
		b = le.AppendUint16(b, uint16(bits>>16))
	}
	b = appendF32(b, p.PreMix)
	per := (c.EngramMaxNgram - 1) * c.EngramHeads
	for _, ids := range p.Hashes {
		if len(ids) != per {
			return nil, errors.New("dsv41: payload does not fit the model")
		}
		for _, v := range ids {
			b = le.AppendUint64(b, uint64(v))
		}
	}
	b = appendF32(b, p.CompKV)
	b = appendF32(b, p.IndexK)
	b = le.AppendUint32(b, uint32(len(p.TopK)))
	for _, v := range p.TopK {
		b = le.AppendUint32(b, uint32(int32(v)))
	}
	if p.Candidates != nil {
		size := c.CandidateBlockSize
		var blocks []uint32
		for i := 0; i < len(p.Candidates); i += size {
			end := min(i+size, len(p.Candidates))
			in := p.Candidates[i]
			for _, v := range p.Candidates[i:end] {
				if v != in {
					return nil, errors.New("dsv41: candidate pool not made of whole blocks")
				}
			}
			if in {
				blocks = append(blocks, uint32(i/size))
			}
		}
		if len(blocks) > c.CandidateBlocks {
			return nil, errors.New("dsv41: candidate pool too large")
		}
		b = le.AppendUint32(b, uint32(len(p.Candidates)))
		b = le.AppendUint32(b, uint32(len(blocks)))
		for _, v := range blocks {
			b = le.AppendUint32(b, v)
		}
	}
	if p.WantMain {
		b = le.AppendUint32(b, uint32(len(p.Main)/c.Dim))
		for _, v := range p.Main {
			bits := math.Float32bits(v)
			if bits&0xffff != 0 {
				return nil, errors.New("dsv41: main stream value not bf16")
			}
			b = le.AppendUint16(b, uint16(bits>>16))
		}
	}
	return b, nil
}

func appendF32(b []byte, v []float32) []byte {
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

var errPayload = errors.New("dsv41: malformed payload")

// DecodePayload reads one Payload from the start of b and returns the
// rest. positions bounds the prompt length and the candidate pool.
func DecodePayload(b []byte, c *Config, positions int) (*Payload, []byte, error) {
	le := binary.LittleEndian
	bad := false
	take := func(n int) []byte {
		if bad || len(b) < n {
			bad = true
			return nil
		}
		out := b[:n]
		b = b[n:]
		return out
	}
	u32 := func() uint32 {
		if x := take(4); x != nil {
			return le.Uint32(x)
		}
		return 0
	}
	f32s := func(n int) []float32 {
		x := take(4 * n)
		if x == nil {
			return nil
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(le.Uint32(x[4*i:]))
		}
		return out
	}
	p := &Payload{}
	prefill := u32()
	p.Source = int(int32(u32()))
	flags := u32()
	if bad || prefill > uint32(positions) || flags&^(hasCompKV|hasIndexK|hasCandidates|hasMain) != 0 {
		return nil, nil, errPayload
	}
	p.Prefill = int(prefill)
	if p.Source != -1 && !c.isKVSource(p.Source) {
		return nil, nil, errPayload
	}
	hd := c.HCMult * c.Dim
	if x := take(2 * hd); x != nil {
		p.H = make([]float32, hd)
		for i := range p.H {
			p.H[i] = math.Float32frombits(uint32(le.Uint16(x[2*i:])) << 16)
		}
	}
	p.PreMix = f32s(c.HCMult)
	per := (c.EngramMaxNgram - 1) * c.EngramHeads
	for range c.EngramLayers {
		x := take(8 * per)
		if x == nil {
			break
		}
		ids := make([]int64, per)
		for i := range ids {
			ids[i] = int64(le.Uint64(x[8*i:]))
		}
		p.Hashes = append(p.Hashes, ids)
	}
	if flags&hasCompKV != 0 {
		p.CompKV = f32s(c.HeadDim)
	}
	if flags&hasIndexK != 0 {
		p.IndexK = f32s(c.IndexDim)
	}
	if n := u32(); n > uint32(c.IndexTopK) {
		bad = true
	} else if n > 0 {
		p.TopK = make([]int, n)
		for i := range p.TopK {
			p.TopK[i] = int(int32(u32()))
			if p.TopK[i] < -1 {
				bad = true
			}
		}
	}
	if flags&hasCandidates != 0 {
		n, k := u32(), u32()
		if bad || n > uint32(positions) || k > uint32(c.CandidateBlocks) {
			return nil, nil, errPayload
		}
		p.Candidates = make([]bool, n)
		size := c.CandidateBlockSize
		prev := -1
		for range k {
			i := int(u32())
			if i <= prev || i*size >= int(n) {
				bad = true
				break
			}
			prev = i
			for t := i * size; t < min((i+1)*size, int(n)); t++ {
				p.Candidates[t] = true
			}
		}
	}
	if flags&hasMain != 0 {
		p.WantMain = true
		n := u32()
		if n > uint32(len(c.DraftTargets)) {
			bad = true
		} else if x := take(2 * int(n) * c.Dim); x != nil {
			p.Main = make([]float32, int(n)*c.Dim)
			for i := range p.Main {
				p.Main[i] = math.Float32frombits(uint32(le.Uint16(x[2*i:])) << 16)
			}
		}
	}
	if bad {
		return nil, nil, errPayload
	}
	return p, b, nil
}

// EncodePayloads encodes the Payloads of consecutive positions.
func EncodePayloads(c *Config, ps []*Payload) ([]byte, error) {
	var b []byte
	for _, p := range ps {
		var err error
		if b, err = AppendPayload(b, c, p); err != nil {
			return nil, err
		}
	}
	return b, nil
}

// DecodePayloads decodes the Payloads of n consecutive positions, which
// must be all of b.
func DecodePayloads(b []byte, c *Config, n, positions int) ([]*Payload, error) {
	out := make([]*Payload, 0, n)
	for range n {
		p, rest, err := DecodePayload(b, c, positions)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
		b = rest
	}
	if len(b) != 0 {
		return nil, errPayload
	}
	return out, nil
}
