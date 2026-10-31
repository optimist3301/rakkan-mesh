package exec

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

// Hash identifies a module by the SHA-256 of its bytes.
type Hash [sha256.Size]byte

func HashOf(module []byte) Hash { return sha256.Sum256(module) }

func (h Hash) String() string { return hex.EncodeToString(h[:]) }

// ParseHash parses a hex module hash.
func ParseHash(s string) (Hash, bool) {
	var h Hash
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != len(h) {
		return h, false
	}
	copy(h[:], b)
	return h, true
}

// ModuleStore keeps modules by hash, least recently used first out. Modules
// are content-addressed, so a stored module is always the one its hash names.
type ModuleStore struct {
	mu       sync.Mutex
	maxBytes int
	size     int
	order    *list.List
	entries  map[Hash]*list.Element
}

type storedModule struct {
	hash Hash
	data []byte
}

func NewModuleStore(maxBytes int) *ModuleStore {
	return &ModuleStore{maxBytes: maxBytes, order: list.New(), entries: map[Hash]*list.Element{}}
}

// Put stores a module and returns its hash.
func (s *ModuleStore) Put(module []byte) Hash {
	h := HashOf(module)
	if len(module) > s.maxBytes {
		return h
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.entries[h]; ok {
		s.order.MoveToFront(e)
		return h
	}
	s.entries[h] = s.order.PushFront(&storedModule{h, module})
	s.size += len(module)
	for s.size > s.maxBytes {
		old := s.order.Remove(s.order.Back()).(*storedModule)
		delete(s.entries, old.hash)
		s.size -= len(old.data)
	}
	return h
}

// Get returns a stored module.
func (s *ModuleStore) Get(h Hash) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[h]
	if !ok {
		return nil, false
	}
	s.order.MoveToFront(e)
	return e.Value.(*storedModule).data, true
}
