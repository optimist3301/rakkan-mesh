package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const commit = "0123456789abcdef0123456789abcdef01234567"

// fakeHub serves a repository: files by name, the large ones (".safetensors")
// as LFS files with their SHA-256. corrupt names a file served with other
// bytes than its digest says.
func fakeHub(t *testing.T, repo string, files map[string]string, corrupt string) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/models/{org}/{name}/revision/{rev}", func(w http.ResponseWriter, r *http.Request) {
		if r.PathValue("org")+"/"+r.PathValue("name") != repo {
			http.NotFound(w, r)
			return
		}
		if strings.HasPrefix(repo, "gated/") && r.Header.Get("Authorization") != "Bearer hf_ok" {
			http.Error(w, "gated", http.StatusUnauthorized)
			return
		}
		var sib []map[string]any
		for name, data := range files {
			s := map[string]any{"rfilename": name, "size": len(data)}
			if strings.HasSuffix(name, ".safetensors") {
				sum := sha256.Sum256([]byte(data))
				s["lfs"] = map[string]any{"sha256": hex.EncodeToString(sum[:]), "size": len(data)}
			}
			sib = append(sib, s)
		}
		json.NewEncoder(w).Encode(map[string]any{"sha": commit, "siblings": sib})
	})
	mux.HandleFunc("GET /{org}/{name}/resolve/{rev}/{file}", func(w http.ResponseWriter, r *http.Request) {
		data, ok := files[r.PathValue("file")]
		if !ok || r.PathValue("rev") != commit {
			http.NotFound(w, r)
			return
		}
		if r.PathValue("file") == corrupt {
			data = strings.ToUpper(data)
		}
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		io.WriteString(w, data)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

// recordingNode stores blobs by content and records the import request.
type recordingNode struct {
	mu      sync.Mutex
	blobs   map[string]string
	import_ map[string]any
}

func startRecordingNode(t *testing.T) (*recordingNode, string) {
	sock := filepath.Join(shortDir(t), "n.sock")
	l, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	n := &recordingNode{blobs: map[string]string{}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/blobs", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		sum := sha256.Sum256(b)
		id := hex.EncodeToString(sum[:])
		n.mu.Lock()
		n.blobs[id] = string(b)
		n.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"id": id, "size": len(b)})
	})
	mux.HandleFunc("POST /v1/llm/models", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		n.mu.Lock()
		n.import_ = req
		n.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"model": "m-1", "chat": true})
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(l)
	t.Cleanup(func() { srv.Close() })
	return n, sock
}

func waitJob(t *testing.T, j *job) jobJSON {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		v := j.view()
		if v.State != "downloading" && v.State != "importing" {
			return v
		}
		if time.Now().After(deadline) {
			t.Fatalf("import still %s", v.State)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func qwenFiles() map[string]string {
	return map[string]string{
		"config.json":                      `{"model_type":"qwen2"}`,
		"generation_config.json":           `{}`,
		"tokenizer.json":                   `{"model":{}}`,
		"tokenizer_config.json":            `{"chat_template":""}`,
		"model-00001-of-00002.safetensors": "weights one",
		"model-00002-of-00002.safetensors": "weights two",
		"model.safetensors.index.json":     `{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}`,
		"README.md":                        "readme",
		"onnx/model.safetensors":           "not this one",
	}
}

func TestImportHF(t *testing.T) {
	files := qwenFiles()
	hub := fakeHub(t, "Qwen/Tiny", files, "")
	node, sock := startRecordingNode(t)
	im := newImporter(hub.URL, sock)

	info, err := im.inspect(t.Context(), "Qwen/Tiny", "main", "")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range info.Files {
		names = append(names, f.Name)
	}
	want := "config.json generation_config.json tokenizer.json tokenizer_config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors"
	if strings.Join(names, " ") != want || !info.Supported || info.Revision != commit || info.ModelType != "qwen2" {
		t.Fatalf("inspect: %v %+v", names, info)
	}
	j, err := im.start(info, "", "q8_0", "")
	if err != nil {
		t.Fatal(err)
	}
	v := waitJob(t, j)
	if v.State != "done" || v.Model != "m-1" || !v.Chat || v.Done != v.Total || v.Name != "Tiny" {
		t.Fatalf("job %+v", v)
	}
	req := node.import_
	id := func(name string) string { s := sha256.Sum256([]byte(files[name])); return hex.EncodeToString(s[:]) }
	if req["format"] != "hf" || req["quant"] != "q8_0" || req["name"] != "Tiny" || req["config"] != id("config.json") ||
		req["tokenizer"] != id("tokenizer.json") || req["generation_config"] != id("generation_config.json") ||
		fmt.Sprint(req["weights"]) != fmt.Sprint([]any{id("model-00001-of-00002.safetensors"), id("model-00002-of-00002.safetensors")}) {
		t.Fatalf("import request %v", req)
	}
	if len(node.blobs) != 6 {
		t.Errorf("%d blobs, want 6", len(node.blobs))
	}
}

func TestImportChecksDigest(t *testing.T) {
	hub := fakeHub(t, "Qwen/Tiny", qwenFiles(), "model-00002-of-00002.safetensors")
	_, sock := startRecordingNode(t)
	im := newImporter(hub.URL, sock)
	info, err := im.inspect(t.Context(), "Qwen/Tiny", "", "")
	if err != nil {
		t.Fatal(err)
	}
	j, _ := im.start(info, "", "", "")
	if v := waitJob(t, j); v.State != "failed" || !strings.Contains(v.Error, "SHA-256") {
		t.Fatalf("job %s: %s", v.State, v.Error)
	}
}

func TestInspectRefuses(t *testing.T) {
	llama := map[string]string{"config.json": `{"model_type":"llama"}`, "tokenizer.json": "{}", "model.safetensors": "w"}
	hub := fakeHub(t, "gated/Llama", llama, "")
	_, sock := startRecordingNode(t)
	im := newImporter(hub.URL, sock)
	for _, c := range []struct{ repo, rev, token, want string }{
		{"not a repo", "", "", "not a repository name"},
		{"a/b", "../x", "", "bad revision"},
		{"gated/Llama", "", "", "gated or private"},
		{"other/Model", "", "", "no such repository"},
	} {
		if _, err := im.inspect(t.Context(), c.repo, c.rev, c.token); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", c.repo, err, c.want)
		}
	}
	if info, err := im.inspect(t.Context(), "gated/Llama", "", "hf_ok"); err != nil || !info.Supported || len(info.Files) != 3 {
		t.Errorf("with a token: %+v, %v", info, err)
	}

	mistral := fakeHub(t, "mistralai/M", map[string]string{"config.json": `{"model_type":"mistral"}`, "tokenizer.json": "{}", "model.safetensors": "w"}, "")
	im = newImporter(mistral.URL, sock)
	info, err := im.inspect(t.Context(), "mistralai/M", "", "")
	if err != nil || info.Supported || info.Note != "mistral" {
		t.Fatalf("mistral: %+v, %v", info, err)
	}
	if _, err := im.start(info, "", "", ""); err == nil {
		t.Error("an unsupported model started")
	}
}

func TestImportDeepSeekGoesToTheNode(t *testing.T) {
	hub := fakeHub(t, "deepseek-ai/V", map[string]string{
		"config.json": `{"text_config":{"model_type":"deepseek_v41_text"}}`, "tokenizer.json": "{}",
		"model-00001.safetensors": "big", "model-00002.safetensors": "bigger"}, "")
	node, sock := startRecordingNode(t)
	im := newImporter(hub.URL, sock)
	info, err := im.inspect(t.Context(), "deepseek-ai/V", "main", "")
	if err != nil || !info.DeepSeek || info.Total != 9 {
		t.Fatalf("%+v, %v", info, err)
	}
	if _, err := im.start(info, "", "q8_0", ""); err == nil {
		t.Error("DeepSeek with a quant started")
	}
	j, _ := im.start(info, "", "", "DS")
	if v := waitJob(t, j); v.State != "done" {
		t.Fatalf("%+v", v)
	}
	if r := node.import_; r["format"] != "hub" || r["repo"] != "deepseek-ai/V" || r["revision"] != commit || r["name"] != "DS" {
		t.Fatalf("import request %v", r)
	}
	if len(node.blobs) != 0 {
		t.Errorf("rakkan-web uploaded %d blobs for DeepSeek", len(node.blobs))
	}
}
