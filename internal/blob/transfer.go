package blob

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
)

// Protocol transfers blobs between members. A stream carries one request:
//
//	{"op": "manifest", "id": ...}      -> status, manifest
//	{"op": "chunks", "chunks": [...]}  -> one frame per chunk, empty if absent
//
// Any member may fetch any blob whose ID it knows. IDs are hashes of the
// content, so knowing one means having been given it.
const Protocol = protocol.ID("/rakkan/blob/1.0.0")

// batchSize is how many chunks one request asks for.
const batchSize = 8

type request struct {
	Op     string   `json:"op"`
	ID     string   `json:"id,omitempty"`
	Chunks []string `json:"chunks,omitempty"`
}

type status struct {
	OK    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

// Service serves stored blobs to members and fetches blobs from them.
type Service struct {
	node  *mesh.Node
	store *Store
	log   *slog.Logger

	mu       sync.Mutex
	inflight map[ID]*fetchCall // concurrent fetches of one blob share work
}

type fetchCall struct {
	done chan struct{}
	err  error
}

// NewService serves the store's blobs on node.
func NewService(node *mesh.Node, store *Store, log *slog.Logger) *Service {
	if log == nil {
		log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	s := &Service{node: node, store: store, log: log, inflight: map[ID]*fetchCall{}}
	node.Handle(Protocol, s.handle)
	return s
}

// Store is the local store.
func (s *Service) Store() *Store { return s.store }

func writeFrame(w io.Writer, b []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func readFrame(r io.Reader, limit int) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if uint64(n) > uint64(limit) {
		return nil, fmt.Errorf("frame of %d bytes exceeds %d", n, limit)
	}
	b := make([]byte, n)
	_, err := io.ReadFull(r, b)
	return b, err
}

func (s *Service) handle(st network.Stream, _ *membership.Cert) {
	defer st.Close()
	st.SetDeadline(time.Now().Add(2 * time.Minute))
	b, err := readFrame(st, 64<<10)
	if err != nil {
		st.Reset()
		return
	}
	var req request
	if json.Unmarshal(b, &req) != nil {
		st.Reset()
		return
	}
	switch req.Op {
	case "manifest":
		id, err := ParseID(req.ID)
		var m *Manifest
		if err == nil {
			var ok bool
			if m, ok = s.store.Manifest(id); !ok {
				err = ErrNotFound
			}
		}
		if err != nil {
			sb, _ := json.Marshal(status{Error: err.Error()})
			writeFrame(st, sb)
			return
		}
		sb, _ := json.Marshal(status{OK: true})
		if writeFrame(st, sb) == nil {
			writeFrame(st, m.Encode())
		}
	case "chunks":
		if len(req.Chunks) > batchSize {
			st.Reset()
			return
		}
		for _, x := range req.Chunks {
			var data []byte
			if h, err := parseHash(x); err == nil {
				data, _ = s.store.ReadChunk(h)
			}
			if writeFrame(st, data) != nil {
				return
			}
		}
	default:
		st.Reset()
	}
}

func parseHash(s string) (Hash, error) {
	id, err := ParseID(s)
	return Hash(id), err
}

// Fetch makes a blob available locally, downloading what is missing from
// providers in parallel. It fails if the blob is larger than maxSize.
// Concurrent fetches of the same blob share one download.
func (s *Service) Fetch(ctx context.Context, id ID, providers []peer.ID, maxSize int64) error {
	if m, ok := s.store.Manifest(id); ok {
		if m.Size > maxSize {
			return fmt.Errorf("%w: %d bytes, limit %d", ErrTooLarge, m.Size, maxSize)
		}
		return nil
	}
	s.mu.Lock()
	if c, ok := s.inflight[id]; ok {
		s.mu.Unlock()
		select {
		case <-c.done:
			if c.err != nil {
				return c.err
			}
			return s.Fetch(ctx, id, providers, maxSize) // re-check size
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	c := &fetchCall{done: make(chan struct{})}
	s.inflight[id] = c
	s.mu.Unlock()

	c.err = s.fetch(ctx, id, providers, maxSize)
	s.mu.Lock()
	delete(s.inflight, id)
	s.mu.Unlock()
	close(c.done)
	return c.err
}

func (s *Service) fetch(ctx context.Context, id ID, providers []peer.ID, maxSize int64) error {
	if len(providers) == 0 {
		return fmt.Errorf("%w: %s, and no providers to fetch it from", ErrNotFound, id)
	}
	var m *Manifest
	var lastErr error
	for _, p := range providers {
		if m, lastErr = s.fetchManifest(ctx, p, id); lastErr == nil {
			break
		}
	}
	if m == nil {
		return fmt.Errorf("blob %s: %w", id, lastErr)
	}
	if m.Size > maxSize {
		return fmt.Errorf("%w: %d bytes, limit %d", ErrTooLarge, m.Size, maxSize)
	}
	// Hold every chunk, present or not, until the manifest is recorded.
	s.store.hold(m.Chunks...)
	defer s.store.release(m.Chunks...)
	if err := s.fetchChunks(ctx, s.store.Missing(m), providers); err != nil {
		return fmt.Errorf("blob %s: %w", id, err)
	}
	_, err := s.store.AddManifest(m)
	return err
}

func (s *Service) fetchManifest(ctx context.Context, p peer.ID, id ID) (*Manifest, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	st, err := s.node.NewStream(ctx, p, Protocol)
	if err != nil {
		return nil, err
	}
	defer st.Close()
	st.SetDeadline(time.Now().Add(30 * time.Second))
	req, _ := json.Marshal(request{Op: "manifest", ID: id.String()})
	if err := writeFrame(st, req); err != nil {
		return nil, err
	}
	b, err := readFrame(st, 64<<10)
	if err != nil {
		return nil, err
	}
	var stt status
	if err := json.Unmarshal(b, &stt); err != nil {
		return nil, err
	}
	if !stt.OK {
		return nil, fmt.Errorf("%s: %s", p.ShortString(), stt.Error)
	}
	b, err = readFrame(st, 20+maxChunks*32)
	if err != nil {
		return nil, err
	}
	m, err := DecodeManifest(b)
	if err != nil {
		return nil, err
	}
	if m.ID() != id {
		return nil, fmt.Errorf("%w: manifest from %s", ErrCorrupt, p.ShortString())
	}
	return m, nil
}

// maxStrikes drops a provider after this many failed batches.
const maxStrikes = 3

// fetchChunks downloads chunks in batches, a few streams per provider at
// once. A batch that fails, or comes back incomplete or corrupt, goes back
// on the queue for another provider; chunks already received are kept.
func (s *Service) fetchChunks(ctx context.Context, missing []Hash, providers []peer.ID) error {
	if len(missing) == 0 {
		return nil
	}
	queue := make(chan []Hash, len(missing)/batchSize+1)
	for i := 0; i < len(missing); i += batchSize {
		queue <- missing[i:min(i+batchSize, len(missing))]
	}
	var (
		mu        sync.Mutex
		strikes   = map[peer.ID]int{}
		remaining = len(missing)
		next      int
		lastErr   error
	)
	// pickProvider returns the next provider not struck out, round-robin.
	pickProvider := func() (peer.ID, bool) {
		mu.Lock()
		defer mu.Unlock()
		for range providers {
			p := providers[next%len(providers)]
			next++
			if strikes[p] < maxStrikes {
				return p, true
			}
		}
		return "", false
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	errc := make(chan error, 1)
	workers := min(2*len(providers), 8, len(queue))
	for range workers {
		wg.Go(func() {
			for {
				var batch []Hash
				select {
				case batch = <-queue:
				case <-ctx.Done():
					return
				}
				p, ok := pickProvider()
				if !ok {
					mu.Lock()
					err := fmt.Errorf("no provider left with the data (last error: %v)", lastErr)
					mu.Unlock()
					select {
					case errc <- err:
					default:
					}
					cancel()
					return
				}
				got, err := s.fetchBatch(ctx, p, batch)
				mu.Lock()
				remaining -= len(got)
				done := remaining == 0
				if err != nil {
					strikes[p]++
					lastErr = err
					s.log.Debug("chunk batch failed", "provider", p.ShortString(), "err", err)
				}
				mu.Unlock()
				if rest := without(batch, got); len(rest) > 0 {
					queue <- rest
				}
				if done {
					cancel()
					return
				}
			}
		})
	}
	wg.Wait()
	select {
	case err := <-errc:
		return err
	default:
	}
	mu.Lock()
	defer mu.Unlock()
	if remaining > 0 {
		if err := ctx.Err(); err != nil && lastErr == nil {
			return err
		}
		return fmt.Errorf("%d chunks not fetched: %v", remaining, lastErr)
	}
	return nil
}

// fetchBatch requests chunks from one provider and stores those that
// verify, returning them.
func (s *Service) fetchBatch(ctx context.Context, p peer.ID, batch []Hash) (got []Hash, err error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	st, err := s.node.NewStream(ctx, p, Protocol)
	if err != nil {
		return nil, err
	}
	defer st.Close()
	stop := context.AfterFunc(ctx, func() { st.Reset() })
	defer stop()
	req := request{Op: "chunks"}
	for _, h := range batch {
		req.Chunks = append(req.Chunks, h.String())
	}
	b, _ := json.Marshal(req)
	if err := writeFrame(st, b); err != nil {
		return nil, err
	}
	for _, h := range batch {
		data, err := readFrame(st, ChunkSize)
		if err != nil {
			return got, err
		}
		if len(data) == 0 {
			return got, fmt.Errorf("%s lacks chunk %s", p.ShortString(), h)
		}
		if err := s.store.PutChunk(h, data); err != nil {
			if errors.Is(err, ErrCorrupt) {
				return got, fmt.Errorf("%w: chunk %s from %s", err, h, p.ShortString())
			}
			return got, err
		}
		got = append(got, h)
	}
	return got, nil
}

func without(all, got []Hash) []Hash {
	if len(got) == 0 {
		return all
	}
	have := map[Hash]bool{}
	for _, h := range got {
		have[h] = true
	}
	var out []Hash
	for _, h := range all {
		if !have[h] {
			out = append(out, h)
		}
	}
	return out
}
