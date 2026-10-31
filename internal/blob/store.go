// Package blob stores and transfers content-addressed data.
//
// A blob is cut into fixed-size chunks, each named by its SHA-256. Its
// manifest lists the blob's size and chunk hashes, and the blob's ID is the
// SHA-256 of the manifest. Every chunk is verified on arrival, so data from
// an untrusted peer is checked before it touches the disk; identical chunks
// are stored once; an interrupted transfer resumes from the chunks already
// held; and chunks can come from several peers at once.
package blob

import (
	"bytes"
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"
)

// ChunkSize is the size of every chunk but a blob's last.
const ChunkSize = 1 << 20

// maxChunks bounds a manifest: 4 TiB at 1 MiB per chunk.
const maxChunks = 1 << 22

var (
	ErrNotFound = errors.New("blob not found")
	ErrCorrupt  = errors.New("blob data does not match its hash")
	ErrTooLarge = errors.New("blob too large")
)

// Hash names a chunk; ID names a blob.
type (
	Hash [sha256.Size]byte
	ID   [sha256.Size]byte
)

func (h Hash) String() string { return hex.EncodeToString(h[:]) }
func (id ID) String() string  { return hex.EncodeToString(id[:]) }
func (id ID) IsZero() bool    { return id == ID{} }

// ParseID parses a hex blob ID.
func ParseID(s string) (ID, error) {
	var id ID
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != len(id) {
		return id, fmt.Errorf("invalid blob ID %q", s)
	}
	copy(id[:], b)
	return id, nil
}

// Manifest describes a blob.
type Manifest struct {
	Size   int64
	Chunks []Hash
}

var manifestMagic = []byte("RKB1")

// Encode returns the canonical encoding the blob ID is the hash of.
func (m *Manifest) Encode() []byte {
	b := make([]byte, 0, 4+8+4+4+len(m.Chunks)*sha256.Size)
	b = append(b, manifestMagic...)
	b = binary.BigEndian.AppendUint64(b, uint64(m.Size))
	b = binary.BigEndian.AppendUint32(b, ChunkSize)
	b = binary.BigEndian.AppendUint32(b, uint32(len(m.Chunks)))
	for _, h := range m.Chunks {
		b = append(b, h[:]...)
	}
	return b
}

// ID is the blob's identifier.
func (m *Manifest) ID() ID { return ID(sha256.Sum256(m.Encode())) }

// DecodeManifest parses and checks a manifest.
func DecodeManifest(b []byte) (*Manifest, error) {
	if len(b) < 20 || !bytes.Equal(b[:4], manifestMagic) {
		return nil, errors.New("malformed manifest")
	}
	size := int64(binary.BigEndian.Uint64(b[4:]))
	chunkSize := binary.BigEndian.Uint32(b[12:])
	n := binary.BigEndian.Uint32(b[16:])
	if chunkSize != ChunkSize || n > maxChunks || len(b) != 20+int(n)*sha256.Size {
		return nil, errors.New("malformed manifest")
	}
	// The chunk count must be exactly what the size implies.
	if size < 0 || int64(n) != (size+ChunkSize-1)/ChunkSize {
		return nil, errors.New("manifest size and chunk count disagree")
	}
	m := &Manifest{Size: size, Chunks: make([]Hash, n)}
	for i := range m.Chunks {
		copy(m.Chunks[i][:], b[20+i*sha256.Size:])
	}
	return m, nil
}

// chunkLen is the length of chunk i of a blob of the given size.
func chunkLen(size int64, i int) int {
	return int(min(ChunkSize, size-int64(i)*ChunkSize))
}

// IDOf computes the ID data would have as a blob, without storing it.
func IDOf(data []byte) ID {
	m := &Manifest{Size: int64(len(data))}
	for off := 0; off < len(data); off += ChunkSize {
		m.Chunks = append(m.Chunks, sha256.Sum256(data[off:min(off+ChunkSize, len(data))]))
	}
	return m.ID()
}

// Store keeps blobs on disk, evicting the least recently used unpinned
// blobs beyond a size budget. It is safe for concurrent use.
type Store struct {
	dir      string
	maxBytes int64

	mu    sync.Mutex
	blobs map[ID]*list.Element // of *entry, front most recently used
	lru   *list.List
	refs  map[Hash]int // manifests (and assemblies) referencing each chunk
	size  int64        // bytes of chunk files
	// partial holds chunks on disk that nothing references: left by an
	// interrupted fetch, kept so a retry resumes. They are evicted first,
	// and expire after partialTTL.
	partial map[Hash]time.Time
}

// partialTTL is how long chunks of an unfinished blob are kept for a retry.
const partialTTL = time.Hour

type entry struct {
	id   ID
	m    *Manifest
	pins int
}

// OpenStore opens or creates a store, dropping chunks no manifest uses.
func OpenStore(dir string, maxBytes int64) (*Store, error) {
	for _, d := range []string{"chunks", "manifests", "tmp"} {
		if err := os.MkdirAll(filepath.Join(dir, d), 0o700); err != nil {
			return nil, err
		}
	}
	s := &Store{dir: dir, maxBytes: maxBytes, blobs: map[ID]*list.Element{}, lru: list.New(), refs: map[Hash]int{}, partial: map[Hash]time.Time{}}
	os.RemoveAll(filepath.Join(dir, "tmp"))
	os.MkdirAll(filepath.Join(dir, "tmp"), 0o700)

	ents, err := os.ReadDir(filepath.Join(dir, "manifests"))
	if err != nil {
		return nil, err
	}
	for _, e := range ents {
		id, err := ParseID(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, "manifests", e.Name()))
		var m *Manifest
		if err == nil {
			m, err = DecodeManifest(b)
		}
		if err != nil || m.ID() != id {
			os.Remove(filepath.Join(dir, "manifests", e.Name()))
			continue
		}
		s.index(id, m)
	}
	// Account for chunks in use; keep recent unreferenced ones (from
	// interrupted fetches) as partial and remove the rest.
	err = filepath.WalkDir(filepath.Join(dir, "chunks"), func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := hex.DecodeString(d.Name())
		var h Hash
		fi, ierr := d.Info()
		if err != nil || len(b) != len(h) || ierr != nil {
			return os.Remove(path)
		}
		copy(h[:], b)
		if s.refs[h] == 0 {
			if time.Since(fi.ModTime()) > partialTTL {
				return os.Remove(path)
			}
			s.partial[h] = fi.ModTime()
		}
		s.size += fi.Size()
		return nil
	})
	if err != nil {
		return nil, err
	}
	s.evict()
	return s, nil
}

func (s *Store) index(id ID, m *Manifest) {
	s.blobs[id] = s.lru.PushBack(&entry{id: id, m: m})
	for _, h := range m.Chunks {
		s.refs[h]++
	}
}

func (s *Store) chunkPath(h Hash) string {
	x := h.String()
	return filepath.Join(s.dir, "chunks", x[:2], x)
}

// writeFile writes data atomically.
func (s *Store) writeFile(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Join(s.dir, "tmp"), "w-*")
	if err != nil {
		return err
	}
	_, err = tmp.Write(data)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp.Name(), path)
	}
	if err != nil {
		os.Remove(tmp.Name())
	}
	return err
}

// hold counts chunks as in use while a blob is assembled (by Put or a
// fetch), so evicting another blob sharing them cannot delete them before
// the manifest exists. release undoes it; chunks nothing uses any more
// become partial.
func (s *Store) hold(hs ...Hash) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, h := range hs {
		s.refs[h]++
		delete(s.partial, h)
	}
}

func (s *Store) release(hs ...Hash) {
	s.mu.Lock()
	now := time.Now()
	for _, h := range hs {
		if s.refs[h]--; s.refs[h] <= 0 {
			delete(s.refs, h)
			if s.HasChunk(h) {
				s.partial[h] = now
			}
		}
	}
	s.mu.Unlock()
	s.evict()
}

// removeChunk deletes a chunk file and its accounting; s.mu must be held.
func (s *Store) removeChunk(h Hash) {
	if fi, err := os.Stat(s.chunkPath(h)); err == nil {
		s.size -= fi.Size()
	}
	os.Remove(s.chunkPath(h))
	delete(s.partial, h)
}

// HasChunk reports whether a chunk is on disk.
func (s *Store) HasChunk(h Hash) bool {
	_, err := os.Stat(s.chunkPath(h))
	return err == nil
}

// PutChunk stores a chunk after checking it matches its hash.
func (s *Store) PutChunk(h Hash, data []byte) error {
	if len(data) > ChunkSize || Hash(sha256.Sum256(data)) != h {
		return ErrCorrupt
	}
	if s.HasChunk(h) {
		return nil
	}
	if err := s.writeFile(s.chunkPath(h), data); err != nil {
		return err
	}
	s.mu.Lock()
	s.size += int64(len(data))
	s.mu.Unlock()
	return nil
}

// ReadChunk returns a chunk's data, verified.
func (s *Store) ReadChunk(h Hash) ([]byte, error) {
	b, err := os.ReadFile(s.chunkPath(h))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if Hash(sha256.Sum256(b)) != h {
		os.Remove(s.chunkPath(h)) // disk corruption: refetch rather than serve
		return nil, ErrCorrupt
	}
	return b, nil
}

// Missing lists a manifest's chunks not on disk, without duplicates.
func (s *Store) Missing(m *Manifest) []Hash {
	seen := map[Hash]bool{}
	var out []Hash
	for _, h := range m.Chunks {
		if !seen[h] && !s.HasChunk(h) {
			out = append(out, h)
		}
		seen[h] = true
	}
	return out
}

// AddManifest records a blob whose chunks are all present.
func (s *Store) AddManifest(m *Manifest) (ID, error) {
	if missing := s.Missing(m); len(missing) > 0 {
		return ID{}, fmt.Errorf("%d chunks missing", len(missing))
	}
	id := m.ID()
	s.mu.Lock()
	_, have := s.blobs[id]
	s.mu.Unlock()
	if have {
		s.touch(id)
		return id, nil
	}
	if err := s.writeFile(filepath.Join(s.dir, "manifests", id.String()), m.Encode()); err != nil {
		return ID{}, err
	}
	s.mu.Lock()
	if _, have := s.blobs[id]; !have {
		s.blobs[id] = s.lru.PushFront(&entry{id: id, m: m})
		for _, h := range m.Chunks {
			s.refs[h]++
		}
	}
	s.mu.Unlock()
	s.evict()
	return id, nil
}

// Put stores a stream as a blob, a chunk at a time.
func (s *Store) Put(r io.Reader) (ID, *Manifest, error) {
	m := &Manifest{}
	buf := make([]byte, ChunkSize)
	defer func() { s.release(m.Chunks...) }()
	for {
		n, err := io.ReadFull(r, buf)
		if n > 0 {
			h := Hash(sha256.Sum256(buf[:n]))
			s.hold(h)
			m.Chunks = append(m.Chunks, h)
			if perr := s.PutChunk(h, buf[:n]); perr != nil {
				return ID{}, nil, perr
			}
			m.Size += int64(n)
			if len(m.Chunks) > maxChunks {
				return ID{}, nil, ErrTooLarge
			}
		}
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		}
		if err != nil {
			return ID{}, nil, err
		}
	}
	id, err := s.AddManifest(m)
	return id, m, err
}

// PutBytes stores data as a blob.
func (s *Store) PutBytes(data []byte) (ID, error) {
	id, _, err := s.Put(bytes.NewReader(data))
	return id, err
}

// Manifest returns a stored blob's manifest.
func (s *Store) Manifest(id ID) (*Manifest, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.blobs[id]
	if !ok {
		return nil, false
	}
	return e.Value.(*entry).m, true
}

// Has reports whether a blob is stored.
func (s *Store) Has(id ID) bool {
	_, ok := s.Manifest(id)
	return ok
}

func (s *Store) touch(id ID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.blobs[id]; ok {
		s.lru.MoveToFront(e)
	}
}

// Pin protects a stored blob from eviction until the returned function is
// called.
func (s *Store) Pin(id ID) (unpin func(), ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, have := s.blobs[id]
	if !have {
		return nil, false
	}
	s.lru.MoveToFront(e)
	e.Value.(*entry).pins++
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			e.Value.(*entry).pins--
			s.mu.Unlock()
			s.evict()
		})
	}, true
}

// evict drops expired partial chunks, then, while over budget, partial
// chunks oldest first and least recently used unpinned blobs.
func (s *Store) evict() {
	s.mu.Lock()
	defer s.mu.Unlock()
	var partial []Hash
	for h, at := range s.partial {
		if time.Since(at) > partialTTL {
			s.removeChunk(h)
		} else {
			partial = append(partial, h)
		}
	}
	slices.SortFunc(partial, func(a, b Hash) int { return s.partial[a].Compare(s.partial[b]) })
	for _, h := range partial {
		if s.size <= s.maxBytes {
			break
		}
		s.removeChunk(h)
	}
	for e := s.lru.Back(); e != nil && s.size > s.maxBytes; {
		prev := e.Prev()
		ent := e.Value.(*entry)
		if ent.pins == 0 {
			s.lru.Remove(e)
			delete(s.blobs, ent.id)
			os.Remove(filepath.Join(s.dir, "manifests", ent.id.String()))
			for _, h := range ent.m.Chunks {
				if s.refs[h]--; s.refs[h] == 0 {
					delete(s.refs, h)
					s.removeChunk(h)
				}
			}
		}
		e = prev
	}
}

// Size is the bytes of chunks on disk.
func (s *Store) Size() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.size
}

// Open returns a reader over a stored blob. It pins the blob until closed.
func (s *Store) Open(id ID) (*Reader, error) {
	unpin, ok := s.Pin(id)
	if !ok {
		return nil, ErrNotFound
	}
	m, _ := s.Manifest(id)
	return &Reader{s: s, m: m, unpin: unpin}, nil
}

// ReadAll returns a blob's contents, refusing blobs larger than max.
func (s *Store) ReadAll(id ID, max int64) ([]byte, error) {
	r, err := s.Open(id)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	if r.Size() > max {
		return nil, fmt.Errorf("%w: %d bytes, limit %d", ErrTooLarge, r.Size(), max)
	}
	b := make([]byte, r.Size())
	_, err = io.ReadFull(r, b)
	return b, err
}

// Reader reads a stored blob; it implements io.ReadSeeker and io.ReaderAt.
type Reader struct {
	s     *Store
	m     *Manifest
	off   int64
	unpin func()

	cached int // index of the chunk in buf, or -1
	buf    []byte
}

func (r *Reader) Size() int64 { return r.m.Size }

func (r *Reader) Close() error {
	r.unpin()
	return nil
}

func (r *Reader) ReadAt(p []byte, off int64) (int, error) {
	n := 0
	for n < len(p) {
		if off >= r.m.Size {
			return n, io.EOF
		}
		i := int(off / ChunkSize)
		if r.buf == nil || r.cached != i {
			b, err := r.s.ReadChunk(r.m.Chunks[i])
			if err != nil {
				return n, err
			}
			if len(b) != chunkLen(r.m.Size, i) {
				return n, ErrCorrupt
			}
			r.buf, r.cached = b, i
		}
		c := copy(p[n:], r.buf[off-int64(i)*ChunkSize:])
		n += c
		off += int64(c)
	}
	return n, nil
}

func (r *Reader) Read(p []byte) (int, error) {
	n, err := r.ReadAt(p, r.off)
	r.off += int64(n)
	if err == io.EOF && n > 0 {
		err = nil
	}
	return n, err
}

func (r *Reader) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		offset += r.off
	case io.SeekEnd:
		offset += r.m.Size
	}
	if offset < 0 {
		return 0, errors.New("negative position")
	}
	r.off = offset
	return offset, nil
}
