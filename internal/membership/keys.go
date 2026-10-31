package membership

import (
	"crypto/rand"
	"errors"
	"fmt"
	"io/fs"
	"os"

	"github.com/libp2p/go-libp2p/core/crypto"
)

// LoadOrCreateKey reads an Ed25519 private key from path, creating it with
// owner-only permissions if it does not exist.
func LoadOrCreateKey(path string) (crypto.PrivKey, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		sk, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			return nil, err
		}
		return sk, SaveKey(path, sk)
	}
	if err != nil {
		return nil, err
	}
	return decodeKey(path, b)
}

// LoadKey reads an existing private key.
func LoadKey(path string) (crypto.PrivKey, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return decodeKey(path, b)
}

func decodeKey(path string, b []byte) (crypto.PrivKey, error) {
	sk, err := crypto.UnmarshalPrivateKey(b)
	if err != nil {
		return nil, fmt.Errorf("key %s: %w", path, err)
	}
	if sk.Type() != crypto.Ed25519 {
		return nil, fmt.Errorf("key %s: not Ed25519", path)
	}
	return sk, nil
}

// SaveKey writes a private key readable only by its owner. It refuses to
// overwrite an existing file.
func SaveKey(path string, sk crypto.PrivKey) error {
	b, err := crypto.MarshalPrivateKey(sk)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
