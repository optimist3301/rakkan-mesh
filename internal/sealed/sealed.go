// Package sealed encrypts a message to the holder of an Ed25519 key.
//
// A network's ID is its root key's peer ID, which embeds the public key, so
// every member can seal a message that only the holder of the root key can
// open: relays, other members and the nodes that store the message see
// nothing but its size. The scheme is an anonymous "sealed box": an
// ephemeral X25519 key agrees a secret with the recipient's key (the
// Ed25519 key mapped to Curve25519), HKDF-SHA256 turns the secret into a key
// and AES-256-GCM encrypts. The sender is not authenticated by the box; put
// a signature inside the message when the recipient must know who wrote it.
package sealed

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"errors"

	"filippo.io/edwards25519"
)

const (
	version  = 1
	keyLen   = 32
	nonceLen = 12
	// Overhead is how many bytes a sealed message is longer than its text.
	Overhead = 1 + keyLen + nonceLen + 16
)

// ErrOpen is returned for anything that is not a message sealed to the key:
// the wrong key, a damaged or truncated envelope, another context.
var ErrOpen = errors.New("sealed: cannot open")

// recipient maps an Ed25519 public key to its X25519 form.
func recipient(pub ed25519.PublicKey) (*ecdh.PublicKey, error) {
	p, err := new(edwards25519.Point).SetBytes(pub)
	if err != nil {
		return nil, err
	}
	return ecdh.X25519().NewPublicKey(p.BytesMontgomery())
}

// private maps an Ed25519 private key to the X25519 key of the same pair.
func private(priv ed25519.PrivateKey) (*ecdh.PrivateKey, error) {
	h := sha512.Sum512(priv.Seed())
	return ecdh.X25519().NewPrivateKey(h[:keyLen]) // clamped by X25519 itself
}

func aead(secret, eph, rcpt, context []byte) (cipher.AEAD, error) {
	salt := append(append([]byte{}, eph...), rcpt...)
	k, err := hkdf.Key(sha256.New, secret, salt, "rakkan-sealed-v1\x00"+string(context), 32)
	if err != nil {
		return nil, err
	}
	b, err := aes.NewCipher(k)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(b)
}

// Seal encrypts msg so that only the holder of pub's private key can open
// it. context binds the envelope to its use (a network ID, say): it must be
// the same to open it, and is not part of the envelope.
func Seal(pub ed25519.PublicKey, msg, context []byte) ([]byte, error) {
	rcpt, err := recipient(pub)
	if err != nil {
		return nil, err
	}
	eph, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	secret, err := eph.ECDH(rcpt)
	if err != nil {
		return nil, err
	}
	a, err := aead(secret, eph.PublicKey().Bytes(), rcpt.Bytes(), context)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, Overhead+len(msg))
	out = append(out, version)
	out = append(out, eph.PublicKey().Bytes()...)
	nonce := make([]byte, nonceLen)
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	out = append(out, nonce...)
	return a.Seal(out, nonce, msg, out[:1+keyLen]), nil
}

// Open decrypts an envelope sealed to priv's public key.
func Open(priv ed25519.PrivateKey, env, context []byte) ([]byte, error) {
	if len(env) < Overhead || env[0] != version {
		return nil, ErrOpen
	}
	me, err := private(priv)
	if err != nil {
		return nil, err
	}
	eph, err := ecdh.X25519().NewPublicKey(env[1 : 1+keyLen])
	if err != nil {
		return nil, ErrOpen
	}
	secret, err := me.ECDH(eph)
	if err != nil {
		return nil, ErrOpen
	}
	a, err := aead(secret, env[1:1+keyLen], me.PublicKey().Bytes(), context)
	if err != nil {
		return nil, err
	}
	msg, err := a.Open(nil, env[1+keyLen:1+keyLen+nonceLen], env[1+keyLen+nonceLen:], env[:1+keyLen])
	if err != nil {
		return nil, ErrOpen
	}
	return msg, nil
}
