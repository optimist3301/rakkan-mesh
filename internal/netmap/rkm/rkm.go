// Package rkm reads and writes the portable .rkm container (spec §7).
package rkm

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"slices"
	"time"

	"github.com/fxamacker/cbor/v2"
	"github.com/klauspost/compress/zstd"
	"github.com/zeebo/blake3"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

const ContainerVersion = 1

var Magic = []byte("\x89RKM\r\n\x1a\n")

type header struct {
	Version  uint   `cbor:"1,keyasint"`
	Created  int64  `cbor:"2,keyasint"`
	Producer string `cbor:"3,keyasint,omitempty"`
}

type trailer struct {
	Count  uint64 `cbor:"1,keyasint"`
	Digest []byte `cbor:"2,keyasint"`
}

type record struct {
	_    struct{} `cbor:",toarray"`
	Body []byte
	Sig  []byte
}

// SetDigest is the order-independent digest of a set of observation ids.
func SetDigest(ids [][32]byte) [32]byte {
	sorted := slices.Clone(ids)
	slices.SortFunc(sorted, func(a, b [32]byte) int { return bytes.Compare(a[:], b[:]) })
	sorted = slices.Compact(sorted)
	h := blake3.New()
	h.Write([]byte("rakkan-set-v1\x00"))
	for _, id := range sorted {
		h.Write(id[:])
	}
	var d [32]byte
	h.Sum(d[:0])
	return d
}

// Write emits observations sorted by id with duplicates removed.
func Write(w io.Writer, observations []*obs.Observation, producer string) error {
	sorted := slices.Clone(observations)
	slices.SortFunc(sorted, func(a, b *obs.Observation) int { return bytes.Compare(a.ID[:], b.ID[:]) })
	sorted = slices.CompactFunc(sorted, func(a, b *obs.Observation) bool { return a.ID == b.ID })

	if _, err := w.Write(Magic); err != nil {
		return err
	}
	zw, err := zstd.NewWriter(w, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	if err != nil {
		return err
	}
	bw := bufio.NewWriter(zw)
	put := func(v any) error {
		b, err := obs.Marshal(v)
		if err != nil {
			return err
		}
		_, err = bw.Write(b)
		return err
	}
	if err := put(header{Version: ContainerVersion, Created: time.Now().UnixMilli(), Producer: producer}); err != nil {
		return err
	}
	ids := make([][32]byte, len(sorted))
	for i, o := range sorted {
		ids[i] = o.ID
		if err := put(record{Body: o.Raw, Sig: o.Sig}); err != nil {
			return err
		}
	}
	d := SetDigest(ids)
	if err := put(trailer{Count: uint64(len(sorted)), Digest: d[:]}); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return zw.Close()
}

// Result of reading a container.
type Result struct {
	Observations []*obs.Observation
	Rejected     []error // records dropped for bad signature / non-canonical body
	Producer     string
	Created      time.Time
	DigestOK     bool // trailer digest matches the ids read (only if nothing rejected)
}

// Read parses a container. Invalid records are reported, not fatal (spec §7).
func Read(r io.Reader) (*Result, error) {
	magic := make([]byte, len(Magic))
	if _, err := io.ReadFull(r, magic); err != nil || !bytes.Equal(magic, Magic) {
		return nil, errors.New("not a rakkan map file (bad magic)")
	}
	zr, err := zstd.NewReader(r)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	dec := obs.NewDecoder(bufio.NewReader(zr))

	var h header
	if err := dec.Decode(&h); err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}
	if h.Version != ContainerVersion {
		return nil, fmt.Errorf("unsupported container version %d", h.Version)
	}
	res := &Result{Producer: h.Producer, Created: time.UnixMilli(h.Created)}
	var tr *trailer
	var ids [][32]byte
	for {
		var raw cbor.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("record %d: %w", len(ids)+len(res.Rejected), err)
		}
		if len(raw) > 0 && raw[0]>>5 == 5 { // CBOR major type 5: map → trailer
			tr = new(trailer)
			if err := obs.Unmarshal(raw, tr); err != nil {
				return nil, fmt.Errorf("trailer: %w", err)
			}
			continue
		}
		var rec record
		if err := obs.Unmarshal(raw, &rec); err != nil {
			res.Rejected = append(res.Rejected, err)
			continue
		}
		o, err := obs.Decode(rec.Body, rec.Sig)
		if err != nil {
			res.Rejected = append(res.Rejected, err)
			continue
		}
		ids = append(ids, o.ID)
		res.Observations = append(res.Observations, o)
	}
	if tr == nil {
		return nil, errors.New("truncated file: missing trailer")
	}
	d := SetDigest(ids)
	res.DigestOK = len(res.Rejected) == 0 && tr.Count == uint64(len(ids)) && bytes.Equal(tr.Digest, d[:])
	return res, nil
}
