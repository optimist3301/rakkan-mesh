// Package membership implements admission to a closed Rakkan network.
//
// A network is identified by its root key: the network ID is the root key's
// peer ID, which embeds the Ed25519 public key, so the ID alone is the trust
// anchor operators configure. The root signs node certificates binding a
// peer ID to roles for a validity period. A node holding the admin role may
// in turn sign member certificates, so admission can be delegated without
// handing out the root key; admins cannot create admins, which bounds chains
// to two links. The root publishes revocations as a signed, sequenced list
// of certificate serials and peer IDs.
//
// The root may open the network to guests, nodes without a certificate
// that prove work on their identity, by signing a policy (see Policy).
//
// A certificate proves membership only together with the connection it
// arrives on: libp2p authenticates the remote peer ID, and the certificate
// subject must match it.
package membership

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"
)

const (
	RoleMember = "member"
	// RoleAdmin may issue member certificates.
	RoleAdmin = "admin"
)

var (
	ErrInvalid = errors.New("invalid certificate")
	ErrExpired = errors.New("certificate not currently valid")
	ErrRevoked = errors.New("certificate revoked")
)

// Domain separation keeps signatures over one kind of object from being
// valid for another.
var (
	certDomain = []byte("rakkan-cert-v1\x00")
	crlDomain  = []byte("rakkan-crl-v1\x00")
)

// maxEncoded bounds encoded certificates and revocation lists accepted from
// the network.
const maxEncoded = 64 << 10

// Network is the trust anchor of a closed network.
type Network struct {
	ID   string
	root crypto.PubKey
}

// ParseNetwork returns the network whose root key has the given peer ID.
func ParseNetwork(id string) (*Network, error) {
	pid, err := peer.Decode(id)
	if err != nil {
		return nil, fmt.Errorf("network ID: %w", err)
	}
	pub, err := pid.ExtractPublicKey()
	if err != nil {
		return nil, fmt.Errorf("network ID does not embed a public key: %w", err)
	}
	if pub.Type() != crypto.Ed25519 {
		return nil, errors.New("network root key must be Ed25519")
	}
	return &Network{ID: id, root: pub}, nil
}

// RootKey is the network's root public key.
func (n *Network) RootKey() crypto.PubKey { return n.root }

// NewRootKey creates the root key of a new network.
func NewRootKey() (crypto.PrivKey, *Network, error) {
	sk, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	id, err := peer.IDFromPrivateKey(sk)
	if err != nil {
		return nil, nil, err
	}
	n, err := ParseNetwork(id.String())
	return sk, n, err
}

// Cert is the signed content of a certificate.
type Cert struct {
	Version   int       `json:"v"`
	Network   string    `json:"network"`
	Subject   peer.ID   `json:"subject"`
	Issuer    peer.ID   `json:"issuer"`
	Serial    string    `json:"serial"`
	Roles     []string  `json:"roles"`
	NotBefore time.Time `json:"not_before"`
	NotAfter  time.Time `json:"not_after"`
}

// HasRole reports whether the certificate grants role.
func (c *Cert) HasRole(role string) bool { return slices.Contains(c.Roles, role) }

// SignedCert is a certificate as stored and transmitted: the exact signed
// bytes, the signature, and the issuer's certificate when the issuer is an
// admin rather than the root.
type SignedCert struct {
	Body   []byte      `json:"body"`
	Sig    []byte      `json:"sig"`
	Parent *SignedCert `json:"parent,omitempty"`
}

// Issue signs c with issuer's key. parent is the issuer's own certificate,
// or nil when the issuer is the root.
func Issue(issuer crypto.PrivKey, parent *SignedCert, c Cert) (*SignedCert, error) {
	id, err := peer.IDFromPrivateKey(issuer)
	if err != nil {
		return nil, err
	}
	c.Version = 1
	c.Issuer = id
	if c.Serial == "" {
		var b [16]byte
		if _, err := rand.Read(b[:]); err != nil {
			return nil, err
		}
		c.Serial = hex.EncodeToString(b[:])
	}
	body, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	sig, err := issuer.Sign(append(slices.Clone(certDomain), body...))
	if err != nil {
		return nil, err
	}
	return &SignedCert{Body: body, Sig: sig, Parent: parent}, nil
}

// Verify checks a certificate and its chain at time now. revoked reports
// whether a serial has been revoked; it may be nil.
func (n *Network) Verify(sc *SignedCert, now time.Time, revoked func(serial string) bool) (*Cert, error) {
	return n.verify(sc, now, revoked, 0)
}

func (n *Network) verify(sc *SignedCert, now time.Time, revoked func(string) bool, depth int) (*Cert, error) {
	if sc == nil {
		return nil, fmt.Errorf("%w: missing", ErrInvalid)
	}
	if depth > 1 {
		return nil, fmt.Errorf("%w: chain too long", ErrInvalid)
	}
	var c Cert
	dec := json.NewDecoder(bytes.NewReader(sc.Body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	switch {
	case c.Version != 1:
		return nil, fmt.Errorf("%w: version %d", ErrInvalid, c.Version)
	case c.Network != n.ID:
		return nil, fmt.Errorf("%w: issued for network %s", ErrInvalid, c.Network)
	case c.Subject == "" || c.Serial == "":
		return nil, fmt.Errorf("%w: missing subject or serial", ErrInvalid)
	case now.Before(c.NotBefore) || !now.Before(c.NotAfter):
		return nil, fmt.Errorf("%w: valid %s to %s", ErrExpired, c.NotBefore.Format(time.RFC3339), c.NotAfter.Format(time.RFC3339))
	}

	var key crypto.PubKey
	if sc.Parent == nil {
		if c.Issuer.String() != n.ID {
			return nil, fmt.Errorf("%w: issuer %s is not the network root", ErrInvalid, c.Issuer)
		}
		key = n.root
	} else {
		parent, err := n.verify(sc.Parent, now, revoked, depth+1)
		if err != nil {
			return nil, fmt.Errorf("issuer certificate: %w", err)
		}
		switch {
		case !parent.HasRole(RoleAdmin):
			return nil, fmt.Errorf("%w: issuer %s is not an admin", ErrInvalid, parent.Subject)
		case parent.Subject != c.Issuer:
			return nil, fmt.Errorf("%w: issuer certificate is for %s, not %s", ErrInvalid, parent.Subject, c.Issuer)
		case c.HasRole(RoleAdmin):
			return nil, fmt.Errorf("%w: only the root may grant %s", ErrInvalid, RoleAdmin)
		case c.NotAfter.After(parent.NotAfter):
			return nil, fmt.Errorf("%w: outlives its issuer's certificate", ErrInvalid)
		}
		if key, err = c.Issuer.ExtractPublicKey(); err != nil {
			return nil, fmt.Errorf("%w: issuer key: %v", ErrInvalid, err)
		}
	}
	ok, err := key.Verify(append(slices.Clone(certDomain), sc.Body...), sc.Sig)
	if err != nil || !ok {
		return nil, fmt.Errorf("%w: bad signature", ErrInvalid)
	}
	if revoked != nil && revoked(c.Serial) {
		return nil, fmt.Errorf("%w: serial %s", ErrRevoked, c.Serial)
	}
	if revoked != nil && revoked(c.Subject.String()) {
		return nil, fmt.Errorf("%w: peer %s", ErrRevoked, c.Subject)
	}
	return &c, nil
}

// DecodeCert parses an encoded certificate.
func DecodeCert(b []byte) (*SignedCert, error) {
	if len(b) > maxEncoded {
		return nil, fmt.Errorf("%w: %d bytes", ErrInvalid, len(b))
	}
	var sc SignedCert
	if err := json.Unmarshal(b, &sc); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return &sc, nil
}

// Encode returns the certificate's file format.
func (sc *SignedCert) Encode() []byte {
	b, _ := json.MarshalIndent(sc, "", "  ")
	return b
}

// LoadCert reads a certificate file.
func LoadCert(path string) (*SignedCert, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeCert(b)
}
