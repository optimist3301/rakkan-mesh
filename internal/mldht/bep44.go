package mldht

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"strconv"
	"sync"
)

// MaxValue is the most bytes an item's value holds: BEP 44 bounds the
// bencoded value to 1000 bytes, and values here are byte strings.
const MaxValue = 1000 - 5

// Item is a mutable item: Value signed by Key with Seq, under Salt. Of two
// items under the same key and salt, the higher Seq wins.
type Item struct {
	Key   ed25519.PublicKey
	Salt  []byte
	Seq   int64
	Value []byte
	Sig   []byte
}

// Target is where the DHT keeps the items of a key and salt.
func Target(key ed25519.PublicKey, salt []byte) [20]byte { return sha1Of(key, salt) }

// signed is what an item's signature covers (BEP 44).
func signed(salt []byte, seq int64, value []byte) []byte {
	var b []byte
	if len(salt) > 0 {
		b = append(b, "4:salt"...)
		b = encode(b, salt)
	}
	b = append(b, "3:seqi"...)
	b = strconv.AppendInt(b, seq, 10)
	b = append(b, "e1:v"...)
	return encode(b, value)
}

// Sign makes an item.
func Sign(key ed25519.PrivateKey, salt []byte, seq int64, value []byte) (*Item, error) {
	if len(value) > MaxValue || len(salt) > 64 {
		return nil, errors.New("mldht: value or salt too long")
	}
	pub := key.Public().(ed25519.PublicKey)
	return &Item{Key: pub, Salt: bytes.Clone(salt), Seq: seq, Value: bytes.Clone(value), Sig: ed25519.Sign(key, signed(salt, seq, value))}, nil
}

// Verify checks the item's signature.
func (it *Item) Verify() error {
	if len(it.Key) != ed25519.PublicKeySize || len(it.Sig) != ed25519.SignatureSize || len(it.Value) > MaxValue {
		return errors.New("mldht: malformed item")
	}
	if !ed25519.Verify(it.Key, signed(it.Salt, it.Seq, it.Value), it.Sig) {
		return errors.New("mldht: bad signature")
	}
	return nil
}

// Get finds the newest item of a key and salt that the nodes closest to
// its target hold, verified, with at least minSeq. It returns the nodes'
// answers too, for a Put after it.
func (c *Client) Get(ctx context.Context, key ed25519.PublicKey, salt []byte, minSeq int64) (*Item, error) {
	it, _, err := c.get(ctx, key, salt, minSeq)
	return it, err
}

var ErrNotFound = errors.New("mldht: no item found")

func (c *Client) get(ctx context.Context, key ed25519.PublicKey, salt []byte, minSeq int64) (*Item, []answer, error) {
	target := Target(key, salt)
	answers, err := c.lookup(ctx, target, "get", func() map[string]any {
		a := map[string]any{"target": target[:]}
		if minSeq > 0 {
			a["seq"] = minSeq
		}
		return a
	})
	if err != nil {
		return nil, nil, err
	}
	var best *Item
	for _, a := range answers {
		v, ok := a.r["v"].([]byte) // values here are byte strings
		seq, hasSeq := intAt(a.r, "seq")
		if !ok || !hasSeq || !bytes.Equal(bytesAt(a.r, "k"), key) {
			continue
		}
		it := &Item{Key: key, Salt: salt, Seq: seq, Value: v, Sig: bytesAt(a.r, "sig")}
		if it.Verify() == nil && seq >= minSeq && (best == nil || seq > best.Seq) {
			best = it
		}
	}
	if best == nil {
		return nil, answers, ErrNotFound
	}
	return best, answers, nil
}

// Put stores an item with the nodes closest to its target, and returns
// how many took it. A node holding a newer item refuses it.
func (c *Client) Put(ctx context.Context, it *Item) (int, error) {
	if err := it.Verify(); err != nil {
		return 0, err
	}
	_, answers, err := c.get(ctx, it.Key, it.Salt, 0)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return 0, err
	}
	var mu sync.Mutex
	var wg sync.WaitGroup
	stored, tried := 0, 0
	var lastErr error
	for _, a := range answers {
		if tried == kClosest {
			break
		}
		if len(a.token) == 0 {
			continue
		}
		tried++
		args := map[string]any{"k": []byte(it.Key), "seq": it.Seq, "sig": it.Sig, "token": a.token, "v": it.Value}
		if len(it.Salt) > 0 {
			args["salt"] = it.Salt
		}
		wg.Go(func() {
			_, err := c.query(ctx, a.addr, "put", args)
			mu.Lock()
			defer mu.Unlock()
			if err == nil {
				stored++
			} else {
				lastErr = err
			}
		})
	}
	wg.Wait()
	if stored == 0 {
		return 0, fmt.Errorf("mldht: no node stored the item: %v", lastErr)
	}
	return stored, nil
}
