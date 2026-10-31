package exec

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// The test binary doubles as the sandbox worker.
func TestMain(m *testing.M) {
	if isolate.IsWorker() {
		if err := isolate.RunWorker(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
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

type testNode struct {
	node  *mesh.Node
	svc   *Service
	pool  *isolate.Pool
	blobs *blob.Service
}

func (tn *testNode) close() {
	tn.svc.Close()
	tn.node.Close()
	if tn.pool != nil {
		tn.pool.Close()
	}
}

// add starts a member; slots > 0 makes it an executor.
func (c *cluster) add(slots int) *testNode { return c.addWith(slots, nil) }

// addWith starts a member offering the given capabilities (nil: all).
func (c *cluster) addWith(slots int, caps []string) *testNode {
	t := c.t
	t.Helper()
	sk, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := peer.IDFromPrivateKey(sk)
	cert, err := membership.Issue(c.root, nil, membership.Cert{
		Network: c.net.ID, Subject: id, Roles: []string{membership.RoleMember},
		NotBefore: c.base.Add(-time.Minute), NotAfter: c.base.Add(time.Hour),
	})
	if err != nil {
		t.Fatal(err)
	}
	node, err := mesh.New(mesh.Config{
		Key: sk, Network: c.net, Cert: cert, Bootstrap: c.boot,
		ListenAddrs:       []string{"/ip4/127.0.0.1/udp/0/quic-v1"},
		AuthTimeout:       2 * time.Second,
		HealthInterval:    300 * time.Millisecond,
		DiscoveryInterval: 500 * time.Millisecond,
		MaxBackoff:        time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := blob.OpenStore(t.TempDir(), 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	tn := &testNode{node: node, blobs: blob.NewService(node, store, nil)}
	cfg := Config{Node: node, Blobs: tn.blobs, LoadInterval: 200 * time.Millisecond, Capabilities: caps}
	if slots > 0 {
		tn.pool, err = isolate.NewPool(isolate.Config{Ceiling: sandbox.DefaultLimits(), Workers: slots, Isolation: isolate.StrictIsolation()})
		if err != nil {
			t.Skipf("isolated workers unavailable: %v", err)
		}
		cfg.Pool, cfg.Slots, cfg.MaxQueue = tn.pool, slots, 1
	}
	if tn.svc, err = New(cfg); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(tn.close)
	if len(c.boot) == 0 {
		c.boot = []peer.AddrInfo{node.AddrInfo()}
	}
	return tn
}

// settle waits until n sees every executor's load.
func settle(t *testing.T, n *testNode, executors ...*testNode) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for {
		loads := n.svc.Loads()
		ok := true
		for _, e := range executors {
			if _, seen := loads[e.node.ID()]; !seen {
				ok = false
			}
		}
		if ok {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("executors' load not seen: %v", loads)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func limits() sandbox.Limits {
	l := sandbox.DefaultLimits()
	l.MaxMemoryPages = 16
	return l
}

func TestRunsOnRemoteExecutorAndSendsModuleOnce(t *testing.T) {
	c := newCluster(t)
	exe := c.add(2)
	req := c.add(0) // requester only
	settle(t, req, exe)

	for i := range 3 {
		in := []byte(fmt.Sprintf("hello %d", i))
		res, err := req.svc.Run(t.Context(), Request{Module: w.Echo(), Input: in, Limits: limits()})
		if err != nil {
			t.Fatal(err)
		}
		if res.Node != exe.node.ID() || !bytes.Equal(res.Output, in) {
			t.Fatalf("ran on %s, output %q", res.Node, res.Output)
		}
	}
	if got := exe.svc.modulesReceived.Load(); got != 1 {
		t.Fatalf("module transferred %d times, want once", got)
	}
}

func TestRunsLocallyWhenAlone(t *testing.T) {
	c := newCluster(t)
	n := c.add(1)
	res, err := n.svc.Run(t.Context(), Request{Module: w.Echo(), Input: []byte("local"), Limits: limits()})
	if err != nil || res.Node != n.node.ID() {
		t.Fatalf("res=%v err=%v", res, err)
	}
}

func TestNoExecutors(t *testing.T) {
	c := newCluster(t)
	n := c.add(0)
	if _, err := n.svc.Run(t.Context(), Request{Module: w.Echo(), Limits: limits()}); !errors.Is(err, ErrNoCapacity) {
		t.Fatalf("got %v, want ErrNoCapacity", err)
	}
}

func TestTaskErrorsCrossTheNetworkAndAreNotRetried(t *testing.T) {
	c := newCluster(t)
	a, b := c.add(1), c.add(1)
	req := c.add(0)
	settle(t, req, a, b)
	lim := limits()
	lim.MaxGas = 10_000
	_, err := req.svc.Run(t.Context(), Request{Module: w.Spin(), Limits: lim})
	var re *RemoteError
	if !errors.Is(err, sandbox.ErrOutOfGas) || !errors.As(err, &re) {
		t.Fatalf("got %v, want a remote ErrOutOfGas", err)
	}
	if got := a.svc.modulesReceived.Load() + b.svc.modulesReceived.Load(); got != 1 {
		t.Fatalf("deterministic failure tried on %d executors", got)
	}
}

func TestFailoverWhenExecutorDies(t *testing.T) {
	c := newCluster(t)
	a, b := c.add(1), c.add(1)
	req := c.add(0)
	settle(t, req, a, b)

	type outcome struct {
		res *Result
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		res, err := req.svc.Run(t.Context(), Request{Module: w.Countdown(400_000_000), Limits: limits()})
		done <- outcome{res, err}
	}()
	// Kill whichever executor picked the task up.
	var victim, survivor *testNode
	deadline := time.Now().Add(10 * time.Second)
	for victim == nil {
		switch {
		case a.svc.inflight.Load() > 0:
			victim, survivor = a, b
		case b.svc.inflight.Load() > 0:
			victim, survivor = b, a
		case time.Now().After(deadline):
			t.Fatal("task never started")
		}
		time.Sleep(5 * time.Millisecond)
	}
	victim.close()

	o := <-done
	if o.err != nil {
		t.Fatal(o.err)
	}
	if o.res.Node != survivor.node.ID() || o.res.Attempts != 2 {
		t.Fatalf("ran on %s after %d attempts, want the survivor after 2", o.res.Node.ShortString(), o.res.Attempts)
	}
}

func TestReplicationOutvotesATamperingExecutor(t *testing.T) {
	c := newCluster(t)
	honest1, honest2, liar := c.add(1), c.add(1), c.add(1)
	liar.svc.tamper = func(r *sandbox.Result) { r.Output = []byte("forged") }
	req := c.add(0)
	settle(t, req, honest1, honest2, liar)

	res, err := req.svc.Run(t.Context(), Request{Module: w.Echo(), Input: []byte("truth"), Limits: limits(), Replicas: 3})
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Output) != "truth" {
		t.Fatalf("accepted output %q", res.Output)
	}
	for _, r := range res.Replicas {
		if r.Agreed == (r.Node == liar.node.ID()) {
			t.Fatalf("replica %s agreed=%v", r.Node.ShortString(), r.Agreed)
		}
	}
}

func TestReplicasAgreeOnGas(t *testing.T) {
	c := newCluster(t)
	e1, e2 := c.add(1), c.add(1)
	req := c.add(0)
	settle(t, req, e1, e2)
	res, err := req.svc.Run(t.Context(), Request{Module: w.Countdown(1_000_000), Limits: limits(), Replicas: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Replicas) != 2 || !res.Replicas[0].Agreed || !res.Replicas[1].Agreed || res.Replicas[0].Node == res.Replicas[1].Node {
		t.Fatalf("replicas: %+v", res.Replicas)
	}
}

func TestWaitsForBusyExecutors(t *testing.T) {
	c := newCluster(t)
	exe := c.add(1) // one slot, one queued
	req := c.add(0)
	settle(t, req, exe)
	var wg sync.WaitGroup
	errs := make(chan error, 6)
	for range 6 {
		wg.Go(func() {
			_, err := req.svc.Run(t.Context(), Request{Module: w.Countdown(20_000_000), Limits: limits()})
			errs <- err
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestMapStopsAtFirstMatch(t *testing.T) {
	c := newCluster(t)
	e1, e2 := c.add(2), c.add(2)
	req := c.add(0)
	settle(t, req, e1, e2)

	const n, target = 400, 137
	inputs := make([][]byte, n)
	for i := range inputs {
		inputs[i] = binary.LittleEndian.AppendUint32(nil, uint32(i))
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	found, reported := -1, 0
	for item := range req.svc.Map(ctx, MapRequest{
		Module: w.Match(target), Inputs: inputs, Limits: limits(),
		StopWhen: func(it MapItem) bool { return it.Err == nil && it.Result.ExitCode == 0 },
	}) {
		reported++
		if item.Err != nil {
			t.Fatalf("input %d: %v", item.Index, item.Err)
		}
		if item.Result.ExitCode == 0 {
			found = item.Index
		}
	}
	if found != target {
		t.Fatalf("found %d, want %d", found, target)
	}
	if reported >= n {
		t.Fatalf("all %d inputs ran despite the early stop", reported)
	}
	t.Logf("stopped after %d of %d inputs", reported, n)
}

func bigLimits() sandbox.Limits {
	l := sandbox.DefaultLimits()
	l.MaxMemoryPages = 256 // 16 MiB
	return l
}

func TestLargeInputAndOutputTravelAsBlobs(t *testing.T) {
	c := newCluster(t)
	exe := c.add(1)
	req := c.add(0)
	settle(t, req, exe)

	in := make([]byte, 6<<20)
	rand.Read(in)
	res, err := req.svc.Run(t.Context(), Request{Module: w.EchoPages(100), Input: in, Limits: bigLimits()})
	if err != nil {
		t.Fatal(err)
	}
	if res.Node != exe.node.ID() || res.OutputBlob.IsZero() || len(res.Output) != 0 {
		t.Fatalf("node %s, output blob %s, inline output %d bytes", res.Node.ShortString(), res.OutputBlob, len(res.Output))
	}
	if !exe.blobs.Store().Has(blob.IDOf(in)) {
		t.Fatal("executor did not receive the input as a blob")
	}
	// The output was fetched eagerly: it survives the executor leaving.
	exe.close()
	out, err := req.blobs.Store().ReadAll(res.OutputBlob, 1<<30)
	if err != nil || !bytes.Equal(out, in) {
		t.Fatalf("output blob: %v", err)
	}
}

func TestInputBlobFromStore(t *testing.T) {
	c := newCluster(t)
	exe := c.add(1)
	req := c.add(0)
	settle(t, req, exe)
	in := bytes.Repeat([]byte("rakkan "), 100_000)
	id, err := req.blobs.Store().PutBytes(in)
	if err != nil {
		t.Fatal(err)
	}
	res, err := req.svc.Run(t.Context(), Request{Module: w.EchoPages(20), InputBlob: id, Limits: bigLimits()})
	if err != nil || !bytes.Equal(res.Output, in) {
		t.Fatalf("err=%v, output %d bytes", err, len(res.Output))
	}
	// Limits apply to blob inputs too.
	lim := bigLimits()
	lim.MaxInputBytes = 1000
	if _, err := req.svc.Run(t.Context(), Request{Module: w.EchoPages(20), InputBlob: id, Limits: lim}); !errors.Is(err, sandbox.ErrInputTooLarge) {
		t.Fatalf("got %v, want ErrInputTooLarge", err)
	}
}

func TestReplicasAgreeOnBlobOutputs(t *testing.T) {
	c := newCluster(t)
	e1, e2 := c.add(1), c.add(1)
	req := c.add(0)
	settle(t, req, e1, e2)
	in := make([]byte, 3<<20)
	rand.Read(in)
	res, err := req.svc.Run(t.Context(), Request{Module: w.EchoPages(60), Input: in, Limits: bigLimits(), Replicas: 2})
	if err != nil {
		t.Fatal(err)
	}
	if res.OutputBlob != blob.IDOf(in) || !res.Replicas[0].Agreed || !res.Replicas[1].Agreed {
		t.Fatalf("output %s, replicas %+v", res.OutputBlob, res.Replicas)
	}
}

func gradientPNG(width, height int) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.NRGBA{uint8(x), uint8(y), 128, 255})
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func TestImageTaskRunsRemotely(t *testing.T) {
	c := newCluster(t)
	exe := c.add(1)
	req := c.add(0)
	settle(t, req, exe)
	res, err := req.svc.Run(t.Context(), Request{Module: w.Thumbnail(256, 256, 1, 85), Input: gradientPNG(1000, 600), Limits: bigLimits()})
	if err != nil {
		t.Fatal(err)
	}
	img, err := jpeg.Decode(bytes.NewReader(res.Output))
	if err != nil || res.ExitCode != 0 {
		t.Fatalf("exit %d, decode %v", res.ExitCode, err)
	}
	if b := img.Bounds(); b.Dx() != 256 || b.Dy() != 153 {
		t.Fatalf("thumbnail %dx%d", b.Dx(), b.Dy())
	}
}

func TestCapabilityAwareScheduling(t *testing.T) {
	c := newCluster(t)
	plain := c.addWith(2, []string{}) // offers no image kernels
	imaging := c.add(1)
	req := c.add(0)
	settle(t, req, plain, imaging)
	for range 4 {
		res, err := req.svc.Run(t.Context(), Request{Module: w.Thumbnail(64, 64, 0, 0), Input: gradientPNG(200, 100), Limits: bigLimits()})
		if err != nil {
			t.Fatal(err)
		}
		if res.Node != imaging.node.ID() {
			t.Fatalf("image task ran on a node without image/1")
		}
	}
	// Plain tasks still use the plain node.
	if _, err := req.svc.Run(t.Context(), Request{Module: w.Echo(), Input: []byte("x"), Limits: limits()}); err != nil {
		t.Fatal(err)
	}

	imaging.close()
	_, err := req.svc.Run(t.Context(), Request{Module: w.Thumbnail(64, 64, 0, 0), Input: gradientPNG(10, 10), Limits: bigLimits()})
	if !errors.Is(err, ErrNoCapacity) || !strings.Contains(err.Error(), `capability "image/1"`) {
		t.Fatalf("got %v, want no capacity naming image/1", err)
	}
}

func TestMapSharesAnAttachedBlob(t *testing.T) {
	c := newCluster(t)
	e1, e2 := c.add(2), c.add(2)
	req := c.add(0)
	settle(t, req, e1, e2)
	shared := make([]byte, 3<<20)
	rand.Read(shared)
	id, _ := req.blobs.Store().PutBytes(shared)

	n := 0
	for item := range req.svc.Map(t.Context(), MapRequest{
		Module: w.BlobEcho(), Inputs: make([][]byte, 8), Blobs: []blob.ID{id}, Limits: bigLimits(),
	}) {
		if item.Err != nil {
			t.Fatal(item.Err)
		}
		if item.Result.OutputBlob != id {
			t.Fatalf("task %d read something else", item.Index)
		}
		n++
	}
	if n != 8 {
		t.Fatalf("%d results", n)
	}
	for _, e := range []*testNode{e1, e2} {
		if !e.blobs.Store().Has(id) {
			t.Fatalf("executor %s did not keep the shared blob", e.node.ID().ShortString())
		}
	}
}
