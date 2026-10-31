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

// An open network admits guests besides certified members: nodes that hold
// no certificate but prove work on their identity (see Proof). Whether the
// network is open, and how much work a guest proves, is set by a policy the
// root signs and nodes spread, so a closed network opens, or closes again,
// without reconfiguring its members. Guests are not trusted the way members
// are: what they compute is verified, and what they may ask of others is
// bounded by the credit they earn (package trust).

// RoleGuest marks a peer admitted by proof of work rather than certificate.
const RoleGuest = "guest"

var policyDomain = []byte("rakkan-policy-v1\x00")

// MaxPowBits bounds the work a policy may demand: beyond it, joining takes
// days.
const MaxPowBits = 40

// Policy is the signed content of a network policy. A policy with a higher
// sequence number replaces earlier ones.
type Policy struct {
	Version  int       `json:"v"`
	Network  string    `json:"network"`
	Seq      uint64    `json:"seq"`
	IssuedAt time.Time `json:"issued_at"`
	// Open admits guests.
	Open bool `json:"open"`
	// PowBits is the work a guest proves: the leading zero bits of its
	// proof hash. Each bit doubles the expected work.
	PowBits int `json:"pow_bits,omitempty"`
}

// SignedPolicy is a policy as stored and transmitted.
type SignedPolicy struct {
	Body []byte `json:"body"`
	Sig  []byte `json:"sig"`
}

// IssuePolicy signs a policy with the root key.
func IssuePolicy(root crypto.PrivKey, p Policy) (*SignedPolicy, error) {
	p.Version = 1
	if err := p.validate(); err != nil {
		return nil, err
	}
	body, err := json.Marshal(p)
	if err != nil {
		return nil, err
	}
	sig, err := root.Sign(append(slices.Clone(policyDomain), body...))
	if err != nil {
		return nil, err
	}
	return &SignedPolicy{Body: body, Sig: sig}, nil
}

func (p *Policy) validate() error {
	if p.PowBits < 0 || p.PowBits > MaxPowBits {
		return fmt.Errorf("policy: proof of work of %d bits, limit %d", p.PowBits, MaxPowBits)
	}
	return nil
}

// VerifyPolicy checks that a policy was signed by the network root.
func (n *Network) VerifyPolicy(s *SignedPolicy) (*Policy, error) {
	if s == nil {
		return nil, errors.New("missing policy")
	}
	ok, err := n.root.Verify(append(slices.Clone(policyDomain), s.Body...), s.Sig)
	if err != nil || !ok {
		return nil, errors.New("policy: bad signature")
	}
	var p Policy
	dec := json.NewDecoder(bytes.NewReader(s.Body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("policy: %w", err)
	}
	if p.Version != 1 || p.Network != n.ID {
		return nil, fmt.Errorf("policy for network %s, version %d", p.Network, p.Version)
	}
	if err := p.validate(); err != nil {
		return nil, err
	}
	return &p, nil
}

// DecodePolicy parses an encoded policy.
func DecodePolicy(b []byte) (*SignedPolicy, error) {
	if len(b) > maxEncoded {
		return nil, fmt.Errorf("policy of %d bytes", len(b))
	}
	var s SignedPolicy
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("policy: %w", err)
	}
	return &s, nil
}

// Encode returns the policy's file format.
func (s *SignedPolicy) Encode() []byte {
	b, _ := json.MarshalIndent(s, "", "  ")
	return b
}

// LoadPolicy reads a policy file.
func LoadPolicy(path string) (*SignedPolicy, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodePolicy(b)
}

// Policies tracks the newest policy a node has seen; without one the
// network is closed. It is safe for concurrent use.
type Policies struct {
	net     *Network
	mu      sync.RWMutex
	current *SignedPolicy
	policy  Policy
}

func NewPolicies(n *Network) *Policies { return &Policies{net: n} }

// Update applies s if it is valid and newer than the current policy, and
// reports whether it did.
func (ps *Policies) Update(s *SignedPolicy) (changed bool, err error) {
	p, err := ps.net.VerifyPolicy(s)
	if err != nil {
		return false, err
	}
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.current != nil && p.Seq <= ps.policy.Seq {
		return false, nil
	}
	ps.current, ps.policy = s, *p
	return true, nil
}

// Current returns the newest signed policy, or nil, and its content.
func (ps *Policies) Current() (*SignedPolicy, Policy) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.current, ps.policy
}
