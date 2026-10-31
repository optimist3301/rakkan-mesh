package membership

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
)

// CRL is the signed content of a revocation list. Lists are cumulative: a
// list with a higher sequence number replaces earlier ones.
type CRL struct {
	Version  int       `json:"v"`
	Network  string    `json:"network"`
	Seq      uint64    `json:"seq"`
	IssuedAt time.Time `json:"issued_at"`
	Revoked  []string  `json:"revoked"` // certificate serials and peer IDs
}

// SignedCRL is a revocation list as stored and transmitted.
type SignedCRL struct {
	Body []byte `json:"body"`
	Sig  []byte `json:"sig"`
}

// IssueCRL signs a revocation list with the root key.
func IssueCRL(root crypto.PrivKey, c CRL) (*SignedCRL, error) {
	c.Version = 1
	body, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	sig, err := root.Sign(append(slices.Clone(crlDomain), body...))
	if err != nil {
		return nil, err
	}
	return &SignedCRL{Body: body, Sig: sig}, nil
}

// VerifyCRL checks that a revocation list was signed by the network root.
func (n *Network) VerifyCRL(s *SignedCRL) (*CRL, error) {
	if s == nil {
		return nil, errors.New("missing revocation list")
	}
	ok, err := n.root.Verify(append(slices.Clone(crlDomain), s.Body...), s.Sig)
	if err != nil || !ok {
		return nil, errors.New("revocation list: bad signature")
	}
	var c CRL
	dec := json.NewDecoder(bytes.NewReader(s.Body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("revocation list: %w", err)
	}
	if c.Version != 1 || c.Network != n.ID {
		return nil, fmt.Errorf("revocation list for network %s, version %d", c.Network, c.Version)
	}
	return &c, nil
}

// DecodeCRL parses an encoded revocation list.
func DecodeCRL(b []byte) (*SignedCRL, error) {
	if len(b) > maxEncoded {
		return nil, fmt.Errorf("revocation list of %d bytes", len(b))
	}
	var s SignedCRL
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("revocation list: %w", err)
	}
	return &s, nil
}

// Encode returns the revocation list's file format.
func (s *SignedCRL) Encode() []byte {
	b, _ := json.MarshalIndent(s, "", "  ")
	return b
}

// LoadCRL reads a revocation list file.
func LoadCRL(path string) (*SignedCRL, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeCRL(b)
}

// Revocations tracks the newest revocation list a node has seen. It is safe
// for concurrent use.
type Revocations struct {
	net     *Network
	mu      sync.RWMutex
	current *SignedCRL
	seq     uint64
	revoked map[string]bool
}

func NewRevocations(n *Network) *Revocations {
	return &Revocations{net: n, revoked: map[string]bool{}}
}

// Update applies s if it is valid and newer than the current list, and
// reports whether it did.
func (r *Revocations) Update(s *SignedCRL) (changed bool, err error) {
	c, err := r.net.VerifyCRL(s)
	if err != nil {
		return false, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.current != nil && c.Seq <= r.seq {
		return false, nil
	}
	set := make(map[string]bool, len(c.Revoked))
	for _, serial := range c.Revoked {
		set[serial] = true
	}
	r.current, r.seq, r.revoked = s, c.Seq, set
	return true, nil
}

// Revoked reports whether a certificate serial or a peer ID is revoked.
func (r *Revocations) Revoked(serial string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.revoked[serial]
}

// Current returns the newest list and its sequence number, or nil.
func (r *Revocations) Current() (*SignedCRL, uint64) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.current, r.seq
}
