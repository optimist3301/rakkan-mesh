package main

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/web"
)

// shortDir is a temporary directory with a path short enough for a unix
// socket (108 bytes on Linux).
func shortDir(t *testing.T) string {
	d, err := os.MkdirTemp("", "rw")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(d) })
	return d
}

// fakeNode serves a node API on a unix socket: a status, and a generation
// that streams two lines with a pause between them.
func fakeNode(t *testing.T, sock string, sawAuth *string) {
	l, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		*sawAuth = r.Header.Get("Authorization")
		io.WriteString(w, `{"id":"x","addrs":[]}`)
	})
	mux.HandleFunc("POST /v1/llm/generate", func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body) // as the node's handler reads the request
		io.WriteString(w, `{"text":"a"}`+"\n")
		w.(http.Flusher).Flush()
		time.Sleep(300 * time.Millisecond)
		io.WriteString(w, `{"done":true}`+"\n")
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(l)
	t.Cleanup(func() { srv.Close() })
}

func TestProxy(t *testing.T) {
	sock := filepath.Join(shortDir(t), "rakkan.sock")
	var auth string
	fakeNode(t, sock, &auth)
	ts := httptest.NewServer(web.Handler(nodeAPI(sock), "tok"))
	defer ts.Close()

	req, _ := http.NewRequest("GET", ts.URL+"/v1/status", nil)
	req.Header.Set("Authorization", "Bearer tok")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != 200 || !strings.Contains(string(body), `"addrs"`) {
		t.Fatalf("status: %d %s", res.StatusCode, body)
	}
	if auth != "" {
		t.Errorf("the page's token reached the node: %q", auth)
	}

	// The first token arrives before the generation ends.
	req, _ = http.NewRequest("POST", ts.URL+"/v1/llm/generate", strings.NewReader("{}"))
	req.Header.Set("Authorization", "Bearer tok")
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	res, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	buf := make([]byte, 64)
	n, _ := res.Body.Read(buf)
	if !strings.Contains(string(buf[:n]), `"text":"a"`) || time.Since(start) > 250*time.Millisecond {
		t.Fatalf("first line %q after %s: not streamed", buf[:n], time.Since(start))
	}
	rest, _ := io.ReadAll(res.Body)
	if !strings.Contains(string(rest), "done") {
		t.Fatalf("rest %q", rest)
	}
}

func TestNodeDown(t *testing.T) {
	sock := filepath.Join(shortDir(t), "none.sock")
	ts := httptest.NewServer(web.Handler(nodeAPI(sock), ""))
	defer ts.Close()
	// httptest's Host is 127.0.0.1:port, a loopback host.
	res, err := http.Get(ts.URL + "/v1/status")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusBadGateway || !strings.Contains(string(body), "node_unreachable") {
		t.Fatalf("%d %s", res.StatusCode, body)
	}
}

func TestFindSocket(t *testing.T) {
	dir := shortDir(t)
	if got, _ := findSocket("/x/rakkan.sock", ""); got != "/x/rakkan.sock" {
		t.Errorf("-api: %q", got)
	}
	cfg := filepath.Join(dir, "node.json")
	os.WriteFile(cfg, []byte(`{"api":"data/rakkan.sock","network":"n"}`), 0o600)
	if got, err := findSocket("", cfg); err != nil || got != filepath.Join(dir, "data", "rakkan.sock") {
		t.Errorf("-config: %q, %v", got, err)
	}
	os.WriteFile(cfg, []byte(`{"network":"n"}`), 0o600)
	if _, err := findSocket("", cfg); err == nil {
		t.Error("a config without api")
	}
	if runtime.GOOS != "linux" {
		return // the rest is where the Linux installer puts the socket
	}
	// The installer's place for a user install.
	t.Setenv("XDG_DATA_HOME", dir)
	t.Chdir(dir)
	if _, err := findSocket("", ""); err == nil || !strings.Contains(err.Error(), "found no node API socket") {
		t.Fatalf("no socket: %v", err)
	}
	os.MkdirAll(filepath.Join(dir, "rakkan"), 0o700)
	sock := filepath.Join(dir, "rakkan", "rakkan.sock")
	l, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if got, err := findSocket("", ""); err != nil || got != sock {
		t.Errorf("found %q, %v; want %s", got, err, sock)
	}
}
