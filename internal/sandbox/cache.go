package sandbox

import (
	"container/list"
	"crypto/sha256"
	"sync"
)

// meteredCache remembers instrumented modules by the hash of the original,
// so a module run repeatedly is instrumented once. Together with wazero's
// compilation cache, keyed by the instrumented bytes, it makes repeated runs
// skip both instrumentation and compilation. Only successes are cached.
type meteredCache struct {
	mu       sync.Mutex
	maxBytes int
	size     int
	order    *list.List // front is most recently used
	entries  map[[sha256.Size]byte]*list.Element
}

type meteredEntry struct {
	key     [sha256.Size]byte
	metered []byte
}

func newMeteredCache(maxBytes int) *meteredCache {
	return &meteredCache{maxBytes: maxBytes, order: list.New(), entries: map[[sha256.Size]byte]*list.Element{}}
}

func (c *meteredCache) get(key [sha256.Size]byte) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(e)
	return e.Value.(*meteredEntry).metered, true
}

func (c *meteredCache) add(key [sha256.Size]byte, metered []byte) {
	if len(metered) > c.maxBytes {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.entries[key]; ok {
		return
	}
	c.entries[key] = c.order.PushFront(&meteredEntry{key, metered})
	c.size += len(metered)
	for c.size > c.maxBytes {
		old := c.order.Remove(c.order.Back()).(*meteredEntry)
		delete(c.entries, old.key)
		c.size -= len(old.metered)
	}
}
