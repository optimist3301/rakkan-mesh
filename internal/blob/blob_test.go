package blob

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
)

func randomBytes(n int) []byte {
	b := make([]byte, n)
	rand.Read(b)
	return b
}

func openStore(t *testing.T, dir string, max int64) *Store {
	t.Helper()
	s, err := OpenStore(dir, max)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestStoreRoundTrip(t *testing.T) {
	s := openStore(t, t.TempDir(), 1<<30)
	for _, size := range []int{0, 1, ChunkSize - 1, ChunkSize, ChunkSize + 1, 3*ChunkSize + 12345} {
		data := randomBytes(size)
		id, err := s.PutBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		if id != IDOf(data) {
			t.Fatalf("size %d: stored ID differs from computed ID", size)
		}
		got, err := s.ReadAll(id, int64(size))
		if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("size %d: read back %d bytes, err %v", size, len(got), err)
		}
	}
}

func TestReaderSeeksAcrossChunks(t *testing.T) {
	s := openStore(t, t.TempDir(), 1<<30)
	data := randomBytes(3*ChunkSize + 100)
	id, _ := s.PutBytes(data)
	r, err := s.Open(id)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	off := int64(ChunkSize - 10)
	r.Seek(off, io.SeekStart)
	buf := make([]byte, ChunkSize+20)
	if _, err := io.ReadFull(r, buf); err != nil || !bytes.Equal(buf, data[off:off+int64(len(buf))]) {
		t.Fatalf("read across chunk boundary: %v", err)
	}
}

func TestCorruptChunkDetected(t *testing.T) {
	dir := t.TempDir()
	s := openStore(t, dir, 1<<30)
	id, _ := s.PutBytes(randomBytes(2 * ChunkSize))
	m, _ := s.Manifest(id)
	os.WriteFile(s.chunkPath(m.Chunks[1]), randomBytes(ChunkSize), 0o600)
	if _, err := s.ReadAll(id, 1<<30); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("got %v, want ErrCorrupt", err)
	}
	if err := s.PutChunk(m.Chunks[0], []byte("not it")); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("PutChunk accepted mismatched data: %v", err)
	}
}

func TestEvictionRespectsPins(t *testing.T) {
	s := openStore(t, t.TempDir(), 3*ChunkSize)
	a, _ := s.PutBytes(randomBytes(ChunkSize))
	unpin, _ := s.Pin(a)
	b, _ := s.PutBytes(randomBytes(ChunkSize))
	c, _ := s.PutBytes(randomBytes(ChunkSize))
	d, _ := s.PutBytes(randomBytes(ChunkSize)) // over budget: b is the oldest unpinned
	if !s.Has(a) || s.Has(b) || !s.Has(c) || !s.Has(d) {
		t.Fatalf("after eviction: a=%v b=%v c=%v d=%v", s.Has(a), s.Has(b), s.Has(c), s.Has(d))
	}
	if s.Size() > 3*ChunkSize {
		t.Fatalf("store holds %d bytes, budget %d", s.Size(), 3*ChunkSize)
	}
	unpin()
	s.PutBytes(randomBytes(ChunkSize))
	if s.Has(a) {
		t.Fatal("unpinned blob kept beyond budget")
	}
}

func TestStoreReopens(t *testing.T) {
	dir := t.TempDir()
	s := openStore(t, dir, 1<<30)
	data := randomBytes(2*ChunkSize + 7)
	id, _ := s.PutBytes(data)
	// An orphan chunk, as an interrupted fetch would leave.
	orphan := randomBytes(100)
	s.PutChunk(Hash(IDOf(orphan)), orphan) // wrong hash: rejected, nothing written
	h := sha(orphan)
	s.PutChunk(h, orphan)

	// A stale orphan, from long ago.
	stale := randomBytes(100)
	s.PutChunk(sha(stale), stale)
	old := time.Now().Add(-2 * partialTTL)
	os.Chtimes(s.chunkPath(sha(stale)), old, old)

	s2 := openStore(t, dir, 1<<30)
	got, err := s2.ReadAll(id, 1<<30)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("after reopen: %v", err)
	}
	if !s2.HasChunk(h) {
		t.Fatal("recent partial chunk dropped on reopening: a retry could not resume")
	}
	if s2.HasChunk(sha(stale)) {
		t.Fatal("stale orphan chunk survived reopening")
	}
	if s2.Size() != int64(len(data)+len(orphan)) {
		t.Fatalf("size %d, want %d", s2.Size(), len(data)+len(orphan))
	}
	if _, err := os.Stat(filepath.Join(dir, "manifests", id.String())); err != nil {
		t.Fatal(err)
	}
}

func TestPartialChunksEvictedFirst(t *testing.T) {
	s := openStore(t, t.TempDir(), 3*ChunkSize)
	kept, _ := s.PutBytes(randomBytes(ChunkSize))
	// Two chunks of a blob whose fetch failed.
	var leftovers []Hash
	for range 2 {
		b := randomBytes(ChunkSize)
		h := sha(b)
		s.hold(h)
		s.PutChunk(h, b)
		leftovers = append(leftovers, h)
	}
	s.release(leftovers...)
	if !s.HasChunk(leftovers[0]) {
		t.Fatal("partial chunks dropped while under budget: a retry could not resume")
	}
	// Going over budget sheds partial chunks before any complete blob.
	s.PutBytes(randomBytes(ChunkSize))
	if !s.Has(kept) || s.HasChunk(leftovers[0]) {
		t.Fatalf("kept=%v partial=%v", s.Has(kept), s.HasChunk(leftovers[0]))
	}
}

func TestDecodeManifestRejectsInconsistency(t *testing.T) {
	m := &Manifest{Size: 3 * ChunkSize, Chunks: make([]Hash, 2)} // needs 3 chunks
	if _, err := DecodeManifest(m.Encode()); err == nil {
		t.Fatal("manifest with too few chunks accepted")
	}
}

// --- transfer between nodes ---

type testNode struct {
	node  *mesh.Node
	blobs *Service
}

type cluster struct {
	t    *testing.T
	root crypto.PrivKey
	net  *membership.Network
	base time.Time
	boot []peer.AddrInfo
}

func newCluster(t *testing.T) *cluster {
	root, n, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	return &cluster{t: t, root: root, net: n, base: time.Now()}
}

func (c *cluster) add() *testNode {
	t := c.t
	sk, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	id, _ := peer.IDFromPrivateKey(sk)
	cert, _ := membership.Issue(c.root, nil, membership.Cert{Network: c.net.ID, Subject: id, Roles: []string{membership.RoleMember},
		NotBefore: c.base.Add(-time.Minute), NotAfter: c.base.Add(time.Hour)})
	node, err := mesh.New(mesh.Config{Key: sk, Network: c.net, Cert: cert, Bootstrap: c.boot,
		ListenAddrs: []string{"/ip4/127.0.0.1/udp/0/quic-v1"}, AuthTimeout: 2 * time.Second,
		HealthInterval: 300 * time.Millisecond, DiscoveryInterval: 500 * time.Millisecond, MaxBackoff: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { node.Close() })
	if len(c.boot) == 0 {
		c.boot = []peer.AddrInfo{node.AddrInfo()}
	}
	return &testNode{node: node, blobs: NewService(node, openStore(t, t.TempDir(), 1<<30), nil)}
}

func (tn *testNode) id() peer.ID { return tn.node.ID() }

// connect waits until a has authenticated with each of others.
func connect(t *testing.T, a *testNode, others ...*testNode) {
	t.Helper()
	for _, o := range others {
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		err := a.node.Connect(ctx, o.node.AddrInfo())
		cancel()
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestFetchFromOneProvider(t *testing.T) {
	c := newCluster(t)
	src, dst := c.add(), c.add()
	connect(t, dst, src)
	data := randomBytes(20<<20 + 333)
	id, _ := src.blobs.Store().PutBytes(data)

	if err := dst.blobs.Fetch(t.Context(), id, []peer.ID{src.id()}, 1<<30); err != nil {
		t.Fatal(err)
	}
	got, err := dst.blobs.Store().ReadAll(id, 1<<30)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("fetched blob differs: %v", err)
	}
	if err := dst.blobs.Fetch(t.Context(), id, []peer.ID{src.id()}, 1<<20); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("size limit not enforced: %v", err)
	}
}

// servedBy counts chunk requests a node answers.
func servedBy(tn *testNode) *atomic.Int64 {
	var n atomic.Int64
	tn.node.Handle(Protocol, func(st network.Stream, c *membership.Cert) {
		n.Add(1)
		tn.blobs.handle(st, c)
	})
	return &n
}

func TestFetchFromSeveralProvidersInParallel(t *testing.T) {
	c := newCluster(t)
	a, b, dst := c.add(), c.add(), c.add()
	connect(t, dst, a, b)
	data := randomBytes(32 << 20)
	id, _ := a.blobs.Store().PutBytes(data)
	b.blobs.Store().PutBytes(data)
	na, nb := servedBy(a), servedBy(b)

	if err := dst.blobs.Fetch(t.Context(), id, []peer.ID{a.id(), b.id()}, 1<<30); err != nil {
		t.Fatal(err)
	}
	if na.Load() < 2 || nb.Load() < 2 {
		t.Fatalf("requests served: a=%d b=%d; want both providers used", na.Load(), nb.Load())
	}
}

func TestCorruptProviderIsRoutedAround(t *testing.T) {
	c := newCluster(t)
	bad, good, dst := c.add(), c.add(), c.add()
	connect(t, dst, bad, good)
	data := randomBytes(8 << 20)
	id, _ := good.blobs.Store().PutBytes(data)
	bad.blobs.Store().PutBytes(data)
	// bad flips a byte in every chunk it serves.
	m, _ := bad.blobs.Store().Manifest(id)
	for _, h := range m.Chunks {
		p := bad.blobs.Store().chunkPath(h)
		b, _ := os.ReadFile(p)
		b[0] ^= 0xff
		os.WriteFile(p, b, 0o600)
	}

	if err := dst.blobs.Fetch(t.Context(), id, []peer.ID{bad.id(), good.id()}, 1<<30); err != nil {
		t.Fatal(err)
	}
	if got, _ := dst.blobs.Store().ReadAll(id, 1<<30); !bytes.Equal(got, data) {
		t.Fatal("corrupt data accepted")
	}
}

func TestFetchResumesAfterProviderDies(t *testing.T) {
	c := newCluster(t)
	flaky, backup, dst := c.add(), c.add(), c.add()
	connect(t, dst, flaky, backup)
	data := randomBytes(24 << 20)
	id, _ := flaky.blobs.Store().PutBytes(data)
	backup.blobs.Store().PutBytes(data)

	// First attempt: only flaky, which dies after a few chunk batches.
	var served atomic.Int64
	flaky.node.Handle(Protocol, func(st network.Stream, cert *membership.Cert) {
		if served.Add(1) == 4 {
			go flaky.node.Close()
		}
		flaky.blobs.handle(st, cert)
	})
	err := dst.blobs.Fetch(t.Context(), id, []peer.ID{flaky.id()}, 1<<30)
	if err == nil {
		t.Fatal("fetch from a dying provider succeeded")
	}
	m, _ := backup.blobs.Store().Manifest(id)
	missing := len(dst.blobs.Store().Missing(m))
	if missing == len(m.Chunks) || missing == 0 {
		t.Fatalf("%d of %d chunks missing after the interruption; want a partial download", missing, len(m.Chunks))
	}
	// Second attempt fetches only what is missing, from the backup.
	nb := servedBy(backup)
	if err := dst.blobs.Fetch(t.Context(), id, []peer.ID{backup.id()}, 1<<30); err != nil {
		t.Fatal(err)
	}
	wantRequests := int64(1 + (missing+batchSize-1)/batchSize) // manifest + batches
	if nb.Load() != wantRequests {
		t.Fatalf("backup served %d requests, want %d (only the missing chunks)", nb.Load(), wantRequests)
	}
	if got, _ := dst.blobs.Store().ReadAll(id, 1<<30); !bytes.Equal(got, data) {
		t.Fatal("resumed blob differs")
	}
}

func sha(b []byte) Hash { return Hash(sha256.Sum256(b)) }

func TestOpenRemoteFetchesOnlyWhatIsRead(t *testing.T) {
	c := newCluster(t)
	src, dst := c.add(), c.add()
	connect(t, dst, src)
	data := randomBytes(40 << 20) // 40 chunks
	id, _ := src.blobs.Store().PutBytes(data)

	r, err := dst.blobs.OpenRemote(t.Context(), id, []peer.ID{src.id()})
	if err != nil {
		t.Fatal(err)
	}
	// Read 100 bytes straddling chunks 20 and 21.
	buf := make([]byte, 100)
	off := int64(21*ChunkSize - 50)
	if n, err := r.ReadAt(buf, off); n != 100 || err != nil || !bytes.Equal(buf, data[off:off+100]) {
		t.Fatalf("read %d, %v", n, err)
	}
	m, _ := src.blobs.Store().Manifest(id)
	fetched := len(m.Chunks) - len(dst.blobs.Store().Missing(m))
	if fetched > 2+readAhead {
		t.Fatalf("fetched %d chunks for a 100-byte read", fetched)
	}
	r.Close()
	if dst.blobs.Store().Has(id) {
		t.Fatal("partially read blob recorded as complete")
	}

	// Reading everything completes the blob in the store.
	r, _ = dst.blobs.OpenRemote(t.Context(), id, []peer.ID{src.id()})
	all := make([]byte, len(data))
	if n, err := r.ReadAt(all, 0); n != len(data) || err != nil || !bytes.Equal(all, data) {
		t.Fatalf("full read %d, %v", n, err)
	}
	r.Close()
	if !dst.blobs.Store().Has(id) {
		t.Fatal("fully read blob not recorded")
	}
}
