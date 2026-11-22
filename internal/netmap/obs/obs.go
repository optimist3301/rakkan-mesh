// Package obs implements the rakkan observation: the immutable, signed unit of
// knowledge a map is made of. See docs/netmap-spec/format.md §2.
package obs

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/fxamacker/cbor/v2"
	"github.com/zeebo/blake3"
)

const SchemaVersion = 1

type Mode uint8

const (
	ModeActive  Mode = 1
	ModePassive Mode = 2
	ModeLocal   Mode = 3
)

func (m Mode) String() string {
	switch m {
	case ModeActive:
		return "active"
	case ModePassive:
		return "passive"
	case ModeLocal:
		return "local"
	}
	return fmt.Sprintf("mode(%d)", uint8(m))
}

// Claim is a statement about the observed entity. Values are always strings.
type Claim struct {
	K string `cbor:"1,keyasint"`
	V string `cbor:"2,keyasint"`
	// C is confidence in per-mille. 0 on the wire means "absent" == 1000.
	C uint16 `cbor:"3,keyasint,omitempty"`
}

// Conf returns the confidence in per-mille, applying the default.
func (c Claim) Conf() uint16 {
	if c.C == 0 {
		return 1000
	}
	return c.C
}

type Vantage struct {
	Iface string `cbor:"1,keyasint,omitempty"`
	MAC   string `cbor:"2,keyasint,omitempty"`
	Net   string `cbor:"3,keyasint,omitempty"`
	Addr  string `cbor:"4,keyasint,omitempty"`
}

func (v *Vantage) empty() bool { return v == nil || *v == Vantage{} }

type Body struct {
	V        uint     `cbor:"0,keyasint"`
	Observer []byte   `cbor:"1,keyasint"`
	Time     int64    `cbor:"2,keyasint"`
	Until    int64    `cbor:"3,keyasint,omitempty"`
	Count    uint32   `cbor:"4,keyasint,omitempty"`
	Mode     Mode     `cbor:"5,keyasint"`
	Method   string   `cbor:"6,keyasint"`
	Vantage  *Vantage `cbor:"7,keyasint,omitempty"`
	Subject  []string `cbor:"8,keyasint"`
	Claims   []Claim  `cbor:"9,keyasint,omitempty"`
}

// Observation is a decoded, verified observation together with its raw bytes.
type Observation struct {
	Body
	Raw []byte   // canonical body bytes; the id is computed over these
	ID  [32]byte //
	Sig []byte
}

func (o *Observation) IDHex() string { return hex.EncodeToString(o.ID[:]) }

func (o *Observation) T() time.Time { return time.UnixMilli(o.Time) }

var (
	encMode cbor.EncMode
	decMode cbor.DecMode
)

func init() {
	var err error
	encMode, err = cbor.CoreDetEncOptions().EncMode()
	if err != nil {
		panic(err)
	}
	decMode, err = cbor.DecOptions{
		DupMapKey:   cbor.DupMapKeyEnforcedAPF,
		IndefLength: cbor.IndefLengthForbidden,
	}.DecMode()
	if err != nil {
		panic(err)
	}
}

func Marshal(v any) ([]byte, error)   { return encMode.Marshal(v) }
func Unmarshal(b []byte, v any) error { return decMode.Unmarshal(b, v) }
func NewDecoder(r interface{ Read([]byte) (int, error) }) *cbor.Decoder {
	return decMode.NewDecoder(r)
}

var (
	idDomain  = []byte("rakkan-obs-v1\x00")
	sigDomain = []byte("rakkan-sig-v1\x00")
)

// Canonicalize sorts and dedups subject and claims, and drops empty vantage.
func (b *Body) Canonicalize() {
	slices.Sort(b.Subject)
	b.Subject = slices.Compact(b.Subject)
	slices.SortFunc(b.Claims, func(x, y Claim) int {
		if c := strings.Compare(x.K, y.K); c != 0 {
			return c
		}
		if c := strings.Compare(x.V, y.V); c != 0 {
			return c
		}
		return int(x.Conf()) - int(y.Conf())
	})
	for i := range b.Claims {
		if b.Claims[i].C == 1000 {
			b.Claims[i].C = 0
		}
	}
	b.Claims = slices.Compact(b.Claims)
	if len(b.Claims) == 0 {
		b.Claims = nil
	}
	if b.Vantage.empty() {
		b.Vantage = nil
	}
}

func (b *Body) validate() error {
	if b.V != SchemaVersion {
		return fmt.Errorf("unsupported schema version %d", b.V)
	}
	if len(b.Observer) != ed25519.PublicKeySize {
		return errors.New("observer must be a 32-byte ed25519 key")
	}
	if b.Mode < ModeActive || b.Mode > ModeLocal {
		return fmt.Errorf("invalid mode %d", b.Mode)
	}
	if b.Method == "" {
		return errors.New("empty method")
	}
	if len(b.Subject) == 0 {
		return errors.New("empty subject")
	}
	for _, s := range b.Subject {
		if !strings.Contains(s, ":") {
			return fmt.Errorf("malformed identifier %q", s)
		}
	}
	for _, c := range b.Claims {
		if c.K == "" || c.C > 1000 {
			return fmt.Errorf("malformed claim %+v", c)
		}
	}
	if b.Until != 0 && b.Until < b.Time {
		return errors.New("until before time")
	}
	return nil
}

func ComputeID(raw []byte) [32]byte {
	h := blake3.New()
	h.Write(idDomain)
	h.Write(raw)
	var id [32]byte
	h.Sum(id[:0])
	return id
}

func sigMessage(id [32]byte) []byte {
	return append(slices.Clone(sigDomain), id[:]...)
}

// Sign canonicalizes the body, encodes, hashes and signs it.
func Sign(b Body, key ed25519.PrivateKey) (*Observation, error) {
	b.V = SchemaVersion
	b.Observer = slices.Clone(key.Public().(ed25519.PublicKey))
	b.Subject = slices.Clone(b.Subject)
	b.Claims = slices.Clone(b.Claims)
	if b.Vantage != nil {
		v := *b.Vantage
		b.Vantage = &v
	}
	b.Canonicalize()
	if err := b.validate(); err != nil {
		return nil, err
	}
	raw, err := encMode.Marshal(&b)
	if err != nil {
		return nil, err
	}
	id := ComputeID(raw)
	return &Observation{Body: b, Raw: raw, ID: id, Sig: ed25519.Sign(key, sigMessage(id))}, nil
}

var (
	ErrBadSignature = errors.New("bad signature")
	ErrNotCanonical = errors.New("body is not canonically encoded")
)

// Decode parses and verifies a raw body + signature (spec §2.3, §2.4).
// For future schema versions the body is kept verbatim and only the signature
// is checked; Body.V tells the caller it was not fully understood.
func Decode(raw, sig []byte) (*Observation, error) {
	var head struct {
		V        uint   `cbor:"0,keyasint"`
		Observer []byte `cbor:"1,keyasint"`
	}
	if err := decMode.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	if len(head.Observer) != ed25519.PublicKeySize {
		return nil, errors.New("observer must be a 32-byte ed25519 key")
	}
	id := ComputeID(raw)
	if !ed25519.Verify(head.Observer, sigMessage(id), sig) {
		return nil, ErrBadSignature
	}
	o := &Observation{Raw: slices.Clone(raw), ID: id, Sig: slices.Clone(sig)}
	if head.V != SchemaVersion {
		o.V, o.Observer = head.V, head.Observer
		return o, nil
	}
	if err := decMode.Unmarshal(raw, &o.Body); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	if err := o.Body.validate(); err != nil {
		return nil, err
	}
	again := o.Body
	again.Subject = slices.Clone(o.Subject)
	again.Claims = slices.Clone(o.Claims)
	again.Canonicalize()
	re, err := encMode.Marshal(&again)
	if err != nil || !bytes.Equal(re, raw) {
		return nil, ErrNotCanonical
	}
	return o, nil
}

// DecodeStored rebuilds an observation that was verified when it entered
// local storage. It skips the signature and canonical-form checks (the
// expensive part of Decode): use it only for data this installation
// verified itself, never for anything arriving from outside.
func DecodeStored(raw, sig []byte) (*Observation, error) {
	o := &Observation{Raw: raw, ID: ComputeID(raw), Sig: sig}
	var head struct {
		V uint `cbor:"0,keyasint"`
	}
	if err := decMode.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	if head.V != SchemaVersion {
		var h struct {
			V        uint   `cbor:"0,keyasint"`
			Observer []byte `cbor:"1,keyasint"`
		}
		decMode.Unmarshal(raw, &h)
		o.V, o.Observer = h.V, h.Observer
		return o, nil
	}
	if err := decMode.Unmarshal(raw, &o.Body); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	return o, nil
}

// Get returns the best value for a claim key in this observation.
func (o *Observation) Get(k string) (string, bool) {
	for _, c := range o.Claims {
		if c.K == k {
			return c.V, true
		}
	}
	return "", false
}
