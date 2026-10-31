package membership

import (
	"crypto/rand"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"
)

func newKey(t *testing.T) (crypto.PrivKey, peer.ID) {
	t.Helper()
	sk, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := peer.IDFromPrivateKey(sk)
	return sk, id
}

// base is a fixed reference so certificates issued in sequence share a
// validity period.
var base = time.Now().Truncate(time.Second)

func cert(n *Network, subject peer.ID, roles ...string) Cert {
	return Cert{Network: n.ID, Subject: subject, Roles: roles, NotBefore: base.Add(-time.Minute), NotAfter: base.Add(time.Hour)}
}

func TestCertificates(t *testing.T) {
	root, n := mustRoot(t)
	adminKey, admin := newKey(t)
	_, member := newKey(t)
	otherKey, other := newKey(t)
	otherRoot, otherNet := mustRoot(t)

	adminCert := must(Issue(root, nil, cert(n, admin, RoleMember, RoleAdmin)))
	memberCert := must(Issue(root, nil, cert(n, member, RoleMember)))
	delegated := must(Issue(adminKey, adminCert, cert(n, other, RoleMember)))
	now := time.Now()

	valid := map[string]*SignedCert{"root-issued": memberCert, "admin-issued": delegated, "admin": adminCert}
	for name, sc := range valid {
		t.Run(name, func(t *testing.T) {
			// Round-trip through the file format.
			sc, err := DecodeCert(sc.Encode())
			if err != nil {
				t.Fatal(err)
			}
			if _, err := n.Verify(sc, now, nil); err != nil {
				t.Fatal(err)
			}
		})
	}

	memberAsIssuer := must(Issue(root, nil, cert(n, other, RoleMember)))
	expired := cert(n, member, RoleMember)
	expired.NotAfter = base.Add(-time.Second)
	long := cert(n, other, RoleMember)
	long.NotAfter = base.Add(48 * time.Hour)
	tampered := *memberCert
	tampered.Body = []byte(string(memberCert.Body[:len(memberCert.Body)-2]) + ` }`)

	invalid := map[string]struct {
		sc   *SignedCert
		want error
	}{
		"other network":          {must(Issue(otherRoot, nil, cert(otherNet, member, RoleMember))), ErrInvalid},
		"self-signed":            {must(Issue(otherKey, nil, cert(n, other, RoleMember))), ErrInvalid},
		"issued by a non-admin":  {must(Issue(otherKey, memberAsIssuer, cert(n, member, RoleMember))), ErrInvalid},
		"admin granting admin":   {must(Issue(adminKey, adminCert, cert(n, other, RoleAdmin))), ErrInvalid},
		"outliving its issuer":   {must(Issue(adminKey, adminCert, long)), ErrInvalid},
		"parent of someone else": {must(Issue(otherKey, adminCert, cert(n, member, RoleMember))), ErrInvalid},
		"expired":                {must(Issue(root, nil, expired)), ErrExpired},
		"tampered body":          {&tampered, ErrInvalid},
		"chain too long":         {must(Issue(otherKey, delegated, cert(n, member, RoleMember))), ErrInvalid},
	}
	for name, c := range invalid {
		t.Run(name, func(t *testing.T) {
			if _, err := n.Verify(c.sc, now, nil); !errors.Is(err, c.want) {
				t.Fatalf("got %v, want %v", err, c.want)
			}
		})
	}
}

func TestRevocation(t *testing.T) {
	root, n := mustRoot(t)
	adminKey, admin := newKey(t)
	_, member := newKey(t)
	adminCert := must(Issue(root, nil, cert(n, admin, RoleAdmin)))
	delegated := must(Issue(adminKey, adminCert, cert(n, member, RoleMember)))
	adminSerial := mustVerify(t, n, adminCert).Serial

	revs := NewRevocations(n)
	crl1 := must(IssueCRL(root, CRL{Network: n.ID, Seq: 1, Revoked: []string{adminSerial}}))
	if changed, err := revs.Update(crl1); err != nil || !changed {
		t.Fatalf("update: changed=%v err=%v", changed, err)
	}
	// Revoking an admin invalidates what it issued.
	if _, err := n.Verify(delegated, time.Now(), revs.Revoked); !errors.Is(err, ErrRevoked) {
		t.Fatalf("got %v, want ErrRevoked", err)
	}

	// Older or equal sequence numbers are ignored.
	crl0 := must(IssueCRL(root, CRL{Network: n.ID, Seq: 1}))
	if changed, _ := revs.Update(crl0); changed {
		t.Fatal("replayed sequence number accepted")
	}
	// Only the root can revoke.
	forged := must(IssueCRL(adminKey, CRL{Network: n.ID, Seq: 2}))
	if _, err := revs.Update(forged); err == nil {
		t.Fatal("revocation list signed by an admin accepted")
	}
	// Round-trip through the file format.
	decoded, err := DecodeCRL(crl1.Encode())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := n.VerifyCRL(decoded); err != nil {
		t.Fatal(err)
	}
}

func TestKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "node.key")
	a, err := LoadOrCreateKey(path)
	if err != nil {
		t.Fatal(err)
	}
	b, err := LoadOrCreateKey(path)
	if err != nil {
		t.Fatal(err)
	}
	if !a.Equals(b) {
		t.Fatal("key changed between loads")
	}
	if err := SaveKey(path, a); err == nil {
		t.Fatal("existing key overwritten")
	}
}

func mustRoot(t *testing.T) (crypto.PrivKey, *Network) {
	t.Helper()
	sk, n, err := NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	return sk, n
}

func mustVerify(t *testing.T, n *Network, sc *SignedCert) *Cert {
	t.Helper()
	c, err := n.Verify(sc, time.Now(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
