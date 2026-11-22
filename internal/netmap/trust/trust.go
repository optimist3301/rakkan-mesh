// Package trust is the local book of observers: how much this installation
// believes each signing key. It is deliberately local and never part of a
// map: who you trust is your decision, not a fact about the network.
package trust

import (
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"strings"
)

type Level int

const (
	Blocked Level = iota // observations are ignored
	Unknown              // default: claims count half, may not override identity
	Trusted              // full weight
	Self                 // this installation's own keys
)

var names = [...]string{"blocked", "unknown", "trusted", "self"}

func (l Level) String() string {
	if l < 0 || int(l) >= len(names) {
		return fmt.Sprintf("level(%d)", int(l))
	}
	return names[l]
}

func (l Level) MarshalText() ([]byte, error) { return []byte(l.String()), nil }

func (l *Level) UnmarshalText(b []byte) error {
	v, err := ParseLevel(string(b))
	*l = v
	return err
}

func ParseLevel(s string) (Level, error) {
	for i, n := range names {
		if strings.EqualFold(s, n) {
			return Level(i), nil
		}
	}
	return Unknown, fmt.Errorf("unknown trust level %q (blocked, unknown, trusted, self)", s)
}

// Entry is one key in the book.
type Entry struct {
	Key   string `json:"key"` // hex ed25519 public key
	Level Level  `json:"level"`
	Name  string `json:"name,omitempty"`
	// Guest marks a key that reached the viewer only through a guest of the
	// network, someone who proved work but holds no certificate. While the
	// level is Unknown such a key counts for less than a member's, and its
	// claims are checked against the members' (package resolve). An explicit
	// Trusted or Blocked decision overrides it.
	Guest bool `json:"guest,omitempty"`
}

// Book maps hex public keys to entries. The zero value trusts nobody
// explicitly: everyone is Unknown.
type Book struct {
	entries map[string]Entry
}

func NewBook(entries ...Entry) *Book {
	b := &Book{entries: map[string]Entry{}}
	for _, e := range entries {
		b.entries[strings.ToLower(e.Key)] = e
	}
	return b
}

// WithSelf marks the given key as Self (the running installation).
func (b *Book) WithSelf(pub ed25519.PublicKey) *Book {
	k := hex.EncodeToString(pub)
	e := b.entries[k]
	e.Key, e.Level = k, Self
	if e.Name == "" {
		e.Name = "this machine"
	}
	b.entries[k] = e
	return b
}

func (b *Book) Get(key string) (Entry, bool) {
	if b == nil {
		return Entry{}, false
	}
	e, ok := b.entries[strings.ToLower(key)]
	return e, ok
}

// Level returns the explicit level of a key, Unknown if absent.
func (b *Book) Level(key string) Level {
	if e, ok := b.Get(key); ok {
		return e.Level
	}
	return Unknown
}

func (b *Book) Entries() []Entry {
	if b == nil {
		return nil
	}
	out := make([]Entry, 0, len(b.entries))
	for _, e := range b.entries {
		out = append(out, e)
	}
	return out
}

// Find resolves a user-given key prefix or name to one entry's key.
// Unknown keys can be named by a hex prefix only if `candidates` contains them.
func Find(query string, book *Book, candidates []string) (string, error) {
	q := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(query), "rkm:"))
	var hits []string
	seen := map[string]bool{}
	add := func(k string) {
		if !seen[k] {
			seen[k] = true
			hits = append(hits, k)
		}
	}
	for _, e := range book.Entries() {
		if strings.EqualFold(e.Name, query) {
			return e.Key, nil
		}
		if len(q) >= 6 && strings.HasPrefix(e.Key, q) {
			add(e.Key)
		}
	}
	for _, k := range candidates {
		if len(q) >= 6 && strings.HasPrefix(k, q) {
			add(k)
		}
	}
	if len(q) == 64 && len(hits) == 0 {
		if _, err := hex.DecodeString(q); err == nil {
			return q, nil
		}
	}
	switch len(hits) {
	case 0:
		return "", fmt.Errorf("no observer key matches %q (give a name or at least 6 hex characters)", query)
	case 1:
		return hits[0], nil
	}
	return "", fmt.Errorf("%q is ambiguous: %d keys match", query, len(hits))
}
