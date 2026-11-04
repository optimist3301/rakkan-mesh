package blob

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/libp2p/go-libp2p/core/peer"
)

// readAhead is how many chunks past a read are fetched with it, so
// sequential readers wait once per batch rather than per chunk.
const readAhead = 4

// Source is a blob being read: complete in the store, or fetched on demand.
type Source interface {
	Size() int64
	ReadAt(p []byte, off int64) (int, error)
	Close() error
}

// OpenRemote returns a reader over a blob that fetches only the chunks
// actually read, from providers, verifying each. A task reading a little of
// a large blob downloads only that much. Closing the reader records the
// blob as complete if every chunk has arrived; otherwise the chunks fetched
// stay for later reads (see partial chunks in Store). Blobs larger than
// maxSize are refused: even unread, a blob's manifest and chunk list cost
// memory in proportion to its size.
func (s *Service) OpenRemote(ctx context.Context, id ID, providers []peer.ID, maxSize int64) (Source, error) {
	if r, err := s.store.Open(id); err == nil {
		if r.Size() > maxSize {
			r.Close()
			return nil, fmt.Errorf("%w: blob %s of %d bytes, limit %d", ErrTooLarge, id, r.Size(), maxSize)
		}
		return r, nil
	}
	var m *Manifest
	var lastErr error
	for _, p := range providers {
		// Any provider's refusal, an oversized manifest included, is only
		// its claim: ask the next.
		if m, lastErr = s.fetchManifest(ctx, p, id, maxSize); lastErr == nil {
			break
		}
	}
	if m == nil {
		if lastErr == nil {
			lastErr = errors.New("no providers")
		}
		return nil, fmt.Errorf("%w: blob %s: %w", ErrNotFound, id, lastErr)
	}
	s.store.hold(m.Chunks...)
	return &remoteReader{svc: s, ctx: ctx, m: m, providers: providers, cached: -1}, nil
}

type remoteReader struct {
	svc       *Service
	ctx       context.Context
	m         *Manifest
	providers []peer.ID

	mu     sync.Mutex
	cached int
	buf    []byte
	closed bool
}

func (r *remoteReader) Size() int64 { return r.m.Size }

func (r *remoteReader) ReadAt(p []byte, off int64) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if off >= r.m.Size {
		return 0, io.EOF
	}
	end := min(off+int64(len(p)), r.m.Size)
	first, last := int(off/ChunkSize), int((end-1)/ChunkSize)
	if err := r.ensure(first, min(last+readAhead, len(r.m.Chunks)-1)); err != nil {
		return 0, err
	}
	n := 0
	for i := first; i <= last; i++ {
		if r.cached != i {
			b, err := r.svc.store.ReadChunk(r.m.Chunks[i])
			if err != nil {
				return n, err
			}
			if len(b) != chunkLen(r.m.Size, i) {
				return n, ErrCorrupt
			}
			r.buf, r.cached = b, i
		}
		start := max(off+int64(n)-int64(i)*ChunkSize, 0)
		n += copy(p[n:end-off], r.buf[start:])
	}
	if int64(n) < int64(len(p)) {
		return n, io.EOF
	}
	return n, nil
}

// ensure fetches the missing chunks among first..last.
func (r *remoteReader) ensure(first, last int) error {
	var missing []Hash
	seen := map[Hash]bool{}
	for i := first; i <= last; i++ {
		h := r.m.Chunks[i]
		if !seen[h] && !r.svc.store.HasChunk(h) {
			missing = append(missing, h)
		}
		seen[h] = true
	}
	if len(missing) == 0 {
		return nil
	}
	if err := r.svc.fetchChunks(r.ctx, missing, r.providers); err != nil {
		return fmt.Errorf("blob %s: %w", r.m.ID(), err)
	}
	return nil
}

func (r *remoteReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	if len(r.svc.store.Missing(r.m)) == 0 {
		r.svc.store.AddManifest(r.m)
	}
	r.svc.store.release(r.m.Chunks...)
	return nil
}
