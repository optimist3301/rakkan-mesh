package sealed

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	for _, n := range []int{0, 1, 1000, 200000} {
		msg := make([]byte, n)
		rand.Read(msg)
		env, err := Seal(pub, msg, []byte("net"))
		if err != nil {
			t.Fatal(err)
		}
		if len(env) != n+Overhead {
			t.Fatalf("envelope is %d bytes for %d", len(env), n)
		}
		got, err := Open(priv, env, []byte("net"))
		if err != nil || !bytes.Equal(got, msg) {
			t.Fatalf("n=%d: %v", n, err)
		}
	}
}

func TestOnlyTheKeyOpens(t *testing.T) {
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	_, other, _ := ed25519.GenerateKey(rand.Reader)
	env, err := Seal(pub, []byte("secret map"), []byte("net"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(env, []byte("secret map")) {
		t.Fatal("plaintext in the envelope")
	}
	if _, err := Open(other, env, []byte("net")); err != ErrOpen {
		t.Fatalf("another key opened it: %v", err)
	}
}

func TestContextAndTampering(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	env, _ := Seal(pub, []byte("x"), []byte("net-a"))
	if _, err := Open(priv, env, []byte("net-b")); err != ErrOpen {
		t.Fatalf("another context opened it: %v", err)
	}
	for i := range env {
		bad := bytes.Clone(env)
		bad[i] ^= 1
		if _, err := Open(priv, bad, []byte("net-a")); err == nil {
			t.Fatalf("flipping byte %d went unnoticed", i)
		}
	}
	for n := 0; n < len(env); n++ {
		if _, err := Open(priv, env[:n], []byte("net-a")); err == nil {
			t.Fatalf("truncation to %d opened", n)
		}
	}
}

func TestFreshEnvelopes(t *testing.T) {
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	a, _ := Seal(pub, []byte("same"), nil)
	b, _ := Seal(pub, []byte("same"), nil)
	if bytes.Equal(a, b) {
		t.Fatal("two seals of one message are identical")
	}
}
