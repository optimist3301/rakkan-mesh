package api

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/exec"
	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

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

// startNode runs a single-node network with an executor and the API.
func startNode(t *testing.T) *http.Client {
	t.Helper()
	root, n, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	sk, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	id, _ := peer.IDFromPrivateKey(sk)
	cert, _ := membership.Issue(root, nil, membership.Cert{Network: n.ID, Subject: id, Roles: []string{membership.RoleMember},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour)})
	node, err := mesh.New(mesh.Config{Key: sk, Network: n, Cert: cert, ListenAddrs: []string{"/ip4/127.0.0.1/udp/0/quic-v1"}})
	if err != nil {
		t.Fatal(err)
	}
	pool, err := isolate.NewPool(isolate.Config{Ceiling: sandbox.DefaultLimits(), Workers: 2, Isolation: isolate.StrictIsolation()})
	if err != nil {
		t.Skipf("isolated workers unavailable: %v", err)
	}
	store, err := blob.OpenStore(t.TempDir(), 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	blobs := blob.NewService(node, store, nil)
	svc, err := exec.New(exec.Config{Node: node, Blobs: blobs, Pool: pool, Slots: 2})
	if err != nil {
		t.Fatal(err)
	}
	sock := filepath.Join(t.TempDir(), "api.sock")
	l, err := Listen(sock)
	if err != nil {
		t.Fatal(err)
	}
	if fi, _ := os.Stat(sock); fi.Mode().Perm() != 0o600 {
		t.Fatalf("socket permissions %v", fi.Mode().Perm())
	}
	ctx, cancel := context.WithCancel(context.Background())
	go (&Server{Service: svc, Blobs: store, Node: node, Network: n.ID}).Serve(ctx, l)
	t.Cleanup(func() { cancel(); svc.Close(); node.Close(); pool.Close() })
	return &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", sock)
	}}}
}

func post(t *testing.T, c *http.Client, path string, body any) (*http.Response, map[string]any) {
	t.Helper()
	var r *bytes.Reader
	if b, ok := body.([]byte); ok {
		r = bytes.NewReader(b)
	} else {
		j, _ := json.Marshal(body)
		r = bytes.NewReader(j)
	}
	res, err := c.Post("http://rakkan"+path, "application/json", r)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if res.Header.Get("Content-Type") == "application/json" {
		json.NewDecoder(res.Body).Decode(&out)
		res.Body.Close()
	}
	return res, out
}

func TestAPI(t *testing.T) {
	c := startNode(t)

	res, up := post(t, c, "/v1/modules", w.Echo())
	if res.StatusCode != 200 || up["hash"] != exec.HashOf(w.Echo()).String() {
		t.Fatalf("upload: %d %v", res.StatusCode, up)
	}
	res, run := post(t, c, "/v1/run", map[string]any{"module_hash": up["hash"], "input": []byte("hi")})
	if res.StatusCode != 200 || run["output"] != "aGk=" || run["attempts"].(float64) != 1 {
		t.Fatalf("run: %d %v", res.StatusCode, run)
	}

	res, fail := post(t, c, "/v1/run", map[string]any{"module": w.Spin(), "limits": map[string]any{"gas": 1000}})
	if e, _ := fail["error"].(map[string]any); res.StatusCode != 422 || e["kind"] != "out_of_gas" {
		t.Fatalf("task error: %d %v", res.StatusCode, fail)
	}
	res, bad := post(t, c, "/v1/run", map[string]any{"module_hash": "00"})
	if res.StatusCode != 400 {
		t.Fatalf("bad request: %d %v", res.StatusCode, bad)
	}

	// Map with an early stop.
	_, up = post(t, c, "/v1/modules", w.Match(7))
	var inputs [][]byte
	for i := range 40 {
		inputs = append(inputs, binary.LittleEndian.AppendUint32(nil, uint32(i)))
	}
	res, _ = post(t, c, "/v1/map", map[string]any{"module_hash": up["hash"], "inputs": inputs, "stop_on_exit_code": 0})
	sc := bufio.NewScanner(res.Body)
	found, done := false, false
	for sc.Scan() {
		var line map[string]any
		json.Unmarshal(sc.Bytes(), &line)
		if line["done"] == true {
			done = true
		} else if line["exit_code"] == 0.0 && line["index"] == 7.0 {
			found = true
		}
	}
	if !found || !done {
		t.Fatalf("map: found=%v done=%v", found, done)
	}

	st, err := c.Get("http://rakkan/v1/status")
	if err != nil {
		t.Fatal(err)
	}
	var status map[string]any
	json.NewDecoder(st.Body).Decode(&status)
	if load := status["load"].(map[string]any); load["slots"] != 2.0 {
		t.Fatalf("status: %v", status)
	}
}

func TestAPIBlobs(t *testing.T) {
	c := startNode(t)
	data := make([]byte, 5<<20+17)
	rand.Read(data)
	res, put := post(t, c, "/v1/blobs", data)
	if res.StatusCode != 200 || put["id"] != blob.IDOf(data).String() || put["size"] != float64(len(data)) {
		t.Fatalf("put: %d %v", res.StatusCode, put)
	}
	id := put["id"].(string)

	// Range requests work, so downloads can resume.
	req, _ := http.NewRequest("GET", "http://rakkan/v1/blobs/"+id, nil)
	req.Header.Set("Range", "bytes=1048570-1048580")
	got, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	part := make([]byte, 11)
	n, _ := got.Body.Read(part)
	if got.StatusCode != http.StatusPartialContent || !bytes.Equal(part[:n], data[1048570:1048570+n]) {
		t.Fatalf("range: %d %x", got.StatusCode, part[:n])
	}

	// A task over the blob; its large output comes back as a blob.
	_, mod := post(t, c, "/v1/modules", w.EchoPages(100))
	res, run := post(t, c, "/v1/run", map[string]any{"module_hash": mod["hash"], "input_blob": id,
		"limits": map[string]any{"memory_mb": 16}})
	if res.StatusCode != 200 || run["output_blob"] != id || run["output_size"] != float64(len(data)) {
		t.Fatalf("run: %d %v", res.StatusCode, run)
	}
	full, _ := c.Get("http://rakkan/v1/blobs/" + id)
	var buf bytes.Buffer
	buf.ReadFrom(full.Body)
	if !bytes.Equal(buf.Bytes(), data) {
		t.Fatal("downloaded blob differs")
	}
	if res, _ := c.Get("http://rakkan/v1/blobs/" + strings.Repeat("0", 64)); res.StatusCode != 404 {
		t.Fatalf("unknown blob: %d", res.StatusCode)
	}
}
