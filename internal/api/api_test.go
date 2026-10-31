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
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

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
	svc, err := exec.New(exec.Config{Node: node, Pool: pool, Slots: 2})
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
	go (&Server{Service: svc, Node: node, Network: n.ID}).Serve(ctx, l)
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
