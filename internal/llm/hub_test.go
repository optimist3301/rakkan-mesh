package llm

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
)

const (
	testRepo = "deepseek-ai/Tiny"
	testRev  = "0123456789abcdef0123456789abcdef01234567"
)

// fakeHub serves a repository as the Hub does: the tree API, the revision
// API and downloads with ranges. It counts the bytes it serves per file,
// and can cut a download short once (cut) or serve wrong bytes (corrupt).
type fakeHub struct {
	files map[string][]byte
	srv   *httptest.Server

	mu      sync.Mutex
	served  map[string]int64
	gets    map[string]int   // downloads started, ranges included
	cut     map[string]int64 // cut the first download of a file after so many bytes
	corrupt map[string]bool
}

func newFakeHub(t *testing.T, files map[string][]byte) *fakeHub {
	h := &fakeHub{files: files, served: map[string]int64{}, gets: map[string]int{}, cut: map[string]int64{}, corrupt: map[string]bool{}}
	h.srv = httptest.NewServer(http.HandlerFunc(h.serve))
	t.Cleanup(h.srv.Close)
	return h
}

func (h *fakeHub) hub() *Hub { return &Hub{Endpoint: h.srv.URL} }

func (h *fakeHub) bytes(name string) int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.served[name]
}

func (h *fakeHub) serve(w http.ResponseWriter, r *http.Request) {
	switch p := r.URL.Path; {
	case p == "/api/models/"+testRepo+"/revision/main":
		json.NewEncoder(w).Encode(map[string]string{"sha": testRev})
	case p == "/api/models/"+testRepo+"/tree/"+testRev:
		var entries []map[string]any
		for name, data := range h.files {
			e := map[string]any{"type": "file", "path": name, "size": len(data)}
			if strings.HasSuffix(name, ".safetensors") {
				sum := sha256.Sum256(data)
				e["lfs"] = map[string]any{"oid": hex.EncodeToString(sum[:]), "size": len(data)}
				e["oid"] = "0000000000000000000000000000000000000000"
			} else {
				g := sha1.New()
				fmt.Fprintf(g, "blob %d\x00", len(data))
				g.Write(data)
				e["oid"] = hex.EncodeToString(g.Sum(nil))
			}
			entries = append(entries, e)
		}
		json.NewEncoder(w).Encode(entries)
	case strings.HasPrefix(p, "/"+testRepo+"/resolve/"+testRev+"/"):
		name := strings.TrimPrefix(p, "/"+testRepo+"/resolve/"+testRev+"/")
		data, ok := h.files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		h.mu.Lock()
		h.gets[name]++
		if h.corrupt[name] {
			data = append([]byte{data[0] ^ 1}, data[1:]...)
		}
		cut, cutting := h.cut[name]
		delete(h.cut, name)
		h.mu.Unlock()
		from := int64(0)
		status := http.StatusOK
		if rg := r.Header.Get("Range"); rg != "" {
			from, _ = strconv.ParseInt(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"), 10, 64)
			status = http.StatusPartialContent
		}
		body := data[from:]
		if cutting {
			body = body[:min(cut, int64(len(body)))]
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(data)-int(from)))
		w.WriteHeader(status)
		w.Write(body)
		// Cut short, the body ends before its length: the server closes
		// the connection and the client reads an unexpected end.
		h.mu.Lock()
		h.served[name] += int64(len(body))
		h.mu.Unlock()
	default:
		http.NotFound(w, r)
	}
}

// writeSafetensors encodes tensors as a safetensors file.
func writeSafetensors(ts map[string]*dsv41.Tensor) []byte {
	names := make([]string, 0, len(ts))
	for n := range ts {
		names = append(names, n)
	}
	sort.Strings(names)
	h := map[string]any{}
	var data []byte
	for _, n := range names {
		t := ts[n]
		h[n] = map[string]any{"dtype": t.DType, "shape": t.Shape, "data_offsets": []int{len(data), len(data) + len(t.Data)}}
		data = append(data, t.Data...)
	}
	hdr, _ := json.Marshal(h)
	for len(hdr)%8 != 0 {
		hdr = append(hdr, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hdr)))
	return append(append(out, hdr...), data...)
}

// tinyRepo lays the fixture out as DeepSeek-V4.1-Flash's repository is:
// a file per layer, the head in two, the Engram tables of layers 1 and 4
// in files of their own with another of the layer's tensors.
func tinyRepo(t *testing.T) map[string][]byte {
	t.Helper()
	ts, err := dsv41.ReadSafetensors(filepath.Join(dsFixture, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := os.ReadFile(filepath.Join(dsFixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	groups := map[string]map[string]*dsv41.Tensor{}
	put := func(file, name string) {
		if groups[file] == nil {
			groups[file] = map[string]*dsv41.Tensor{}
		}
		groups[file][name] = ts[name]
	}
	for name := range ts {
		l, isLayer := dsv41.TensorLayer(name)
		switch {
		case name == "embed.weight":
			put("model-00001-of-00012.safetensors", name)
		case dsv41.IsHeadTensor(name):
			put("model-00010-of-00012.safetensors", name)
		case isLayer && (l == 1 || l == 4) && (strings.Contains(name, ".engram.embed.") || strings.HasSuffix(name, ".engram.q_weight")):
			put(fmt.Sprintf("model-%05d-of-00012.safetensors", 11+l/4), name)
		case isLayer:
			put(fmt.Sprintf("model-%05d-of-00012.safetensors", 2+l), name)
		default:
			put("model-00010-of-00012.safetensors", name) // DSpark's
		}
	}
	files := map[string][]byte{"config.json": cfg, "tokenizer.json": tinyTokenizer(t)}
	weightMap := map[string]string{}
	for file, g := range groups {
		files[file] = writeSafetensors(g)
		for name := range g {
			weightMap[name] = file
		}
	}
	files["model.safetensors.index.json"], _ = json.Marshal(map[string]any{"weight_map": weightMap})
	return files
}

// The import pins every file and downloads the small ones only.
func TestImportHubDownloadsNoWeights(t *testing.T) {
	files := tinyRepo(t)
	h := newFakeHub(t, files)
	c := newCluster(t)
	coord := c.add(false)
	id, m, err := ImportHub(t.Context(), coord.blobs.Store(), h.hub(), testRepo, "main", "tiny")
	if err != nil {
		t.Fatal(err)
	}
	if m.Hub.Revision != testRev || len(m.Hub.Layers) != 8 || id.IsZero() {
		t.Fatalf("manifest %+v", m.Hub)
	}
	for name := range files {
		if strings.HasSuffix(name, ".safetensors") && h.bytes(name) > 0 {
			t.Errorf("%s downloaded by the import", name)
		}
	}
	if got := m.Hub.Layers[1]; len(got) != 2 {
		t.Errorf("layer 1's files %v: its own and the Engram table's", got)
	}
	// Pins the hashes: a manifest with another hash is another model.
	f := m.Hub.Files[m.Hub.Layers[0][0]]
	if sum := sha256.Sum256(files[m.Hub.Layers[0][0]]); f.SHA256 != hex.EncodeToString(sum[:]) || f.Size != int64(len(files[m.Hub.Layers[0][0]])) {
		t.Errorf("pin %+v", f)
	}
}

// Over a pipeline, each node gets only its layers' files, once, and the
// text is one process's. A download cut short resumes where it stopped.
func TestHubModelOverNodes(t *testing.T) {
	files := tinyRepo(t)
	h := newFakeHub(t, files)
	c := newCluster(t)
	coord := c.addOpts(nodeOpts{hub: h.hub()})
	a := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 1, hub: h.hub()})
	b := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 1, hub: h.hub()})
	model, m, err := ImportHub(t.Context(), coord.blobs.Store(), h.hub(), testRepo, testRev, "tiny")
	if err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.cut[m.Hub.Layers[2][0]] = 1000 // layer 2's file breaks off once
	h.mu.Unlock()
	waitServers(t, coord, 2)
	want := localDeepSeek(t, files["tokenizer.json"], dsPrompt, 12)
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 12, Stages: 2}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	t.Logf("stages %+v", stats.Stages)
	for name, data := range files {
		if !strings.HasSuffix(name, ".safetensors") {
			continue
		}
		served, size := h.bytes(name), int64(len(data))
		switch {
		case name == m.Hub.Layers[2][0] && (served != size || h.gets[name] != 2):
			t.Errorf("%s: %d bytes in %d requests for %d, cut after 1000: not resumed", name, served, h.gets[name], size)
		case name != m.Hub.Layers[2][0] && served > size:
			t.Errorf("%s: %d bytes served for %d: downloaded twice", name, served, size)
		}
	}
	// Each serving node holds just its layers' files.
	for _, n := range []*llmNode{a, b} {
		offered := n.llm.offeredFiles()
		var layers []int
		for i := range m.Hub.Layers {
			if _, ok := offered[m.Hub.Files[m.Hub.Layers[i][0]].key()]; ok {
				layers = append(layers, i)
			}
		}
		t.Logf("%s holds the files of layers %v", n.node.ID().ShortString(), layers)
		if len(layers) == 0 || len(layers) == 8 {
			t.Errorf("a node holds the files of layers %v", layers)
		}
	}

	// The draft model from its files too (from the coordinator's Hub client): the same text.
	drafted, _ := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 12, Stages: 2, Draft: true}, nil)
	if !equalTokens(drafted, want) {
		t.Fatalf("with the draft %v, without %v", drafted, want)
	}

	// A node joining later gets the files from the members, not the Hub.
	before := map[string]int64{}
	for name := range files {
		before[name] = h.bytes(name)
	}
	late := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 1, hub: h.hub()})
	waitServers(t, late, 2)
	deadline := time.Now().Add(10 * time.Second)
	for len(late.llm.servers()[a.node.ID()].Files) == 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	for i := range m.Hub.Layers {
		if _, err := late.llm.hubFile(t.Context(), m.Hub, m.Hub.Layers[i][0]); err != nil {
			t.Fatal(err)
		}
	}
	for name := range files {
		if h.bytes(name) != before[name] {
			t.Errorf("%s downloaded from the Hub again, members had it", name)
		}
	}
}

// Bytes not the pinned ones are refused, from the Hub or from a member.
func TestHubFilesChecked(t *testing.T) {
	files := tinyRepo(t)
	h := newFakeHub(t, files)
	c := newCluster(t)
	liar := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 1})
	n := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 1, hub: h.hub()})
	_, m, err := ImportHub(t.Context(), n.blobs.Store(), h.hub(), testRepo, testRev, "tiny")
	if err != nil {
		t.Fatal(err)
	}
	name := m.Hub.Layers[3][0]
	// The Hub serving wrong bytes: no file.
	h.mu.Lock()
	h.corrupt[name] = true
	h.mu.Unlock()
	if _, err := n.llm.hubFile(t.Context(), m.Hub, name); err == nil || !strings.Contains(err.Error(), "hash") {
		t.Fatalf("corrupt download: %v", err)
	}
	// A member offering other bytes under the file's hash: refused, and
	// the Hub, serving it right again, gives the file.
	h.mu.Lock()
	h.corrupt[name] = false
	h.mu.Unlock()
	other, err := liar.blobs.Store().PutBytes(files[m.Hub.Layers[4][0]])
	if err != nil {
		t.Fatal(err)
	}
	liar.llm.src.mu.Lock()
	liar.llm.src.have[m.Hub.Files[name].key()] = other
	liar.llm.src.mu.Unlock()
	waitServers(t, n, 1)
	deadline := time.Now().Add(10 * time.Second)
	for len(n.llm.servers()[liar.node.ID()].Files) == 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	before := h.bytes(name)
	if _, err := n.llm.hubFile(t.Context(), m.Hub, name); err != nil {
		t.Fatal(err)
	}
	if h.bytes(name) == before {
		t.Fatal("the liar's bytes were taken")
	}
}

// TestImportHubOnline imports DeepSeek-V4.1-Flash from the real Hub (a
// few MiB: config, index, tokenizer) and checks the pins against the
// reference tools' (tools/llmref/dsv41/files.json). RAKKAN_HUB_ONLINE=1.
func TestImportHubOnline(t *testing.T) {
	if os.Getenv("RAKKAN_HUB_ONLINE") == "" {
		t.Skip("RAKKAN_HUB_ONLINE not set")
	}
	var pins struct {
		Repo     string `json:"repo"`
		Revision string `json:"revision"`
		Files    map[string]struct {
			Size   int64  `json:"size"`
			SHA256 string `json:"sha256"`
			SHA1   string `json:"git_blob_sha1"`
		} `json:"files"`
	}
	b, err := os.ReadFile("../../tools/llmref/dsv41/files.json")
	if err != nil {
		t.Fatal(err)
	}
	json.Unmarshal(b, &pins)
	c := newCluster(t)
	n := c.add(false)
	id, m, err := ImportHub(t.Context(), n.blobs.Store(), &Hub{}, pins.Repo, pins.Revision, "DeepSeek-V4.1-Flash")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(m)
	t.Logf("model %s: manifest of %d bytes, %d files pinned", id, len(data), len(m.Hub.Files))
	var total int64
	for name, f := range m.Hub.Files {
		p := pins.Files[name]
		if f.Size != p.Size || f.SHA256 != p.SHA256 || f.SHA1 != p.SHA1 {
			t.Errorf("%s: pinned %+v, the reference tools have %+v", name, f, p)
		}
		total += f.Size
	}
	t.Logf("layer 1's files %v, head's %v; %.1f GiB in all", m.Hub.Layers[1], m.Hub.Head, float64(total)/(1<<30))
	if len(data) > maxManifest/4 {
		t.Errorf("manifest of %d bytes", len(data))
	}
}

// TestHubHeadOnline builds a coordinator's part of DeepSeek-V4.1-Flash from
// the real Hub: 2.5 GiB of two files downloaded, checked against their
// pinned SHA-256 as they arrive, the tensors read from them in place.
// RAKKAN_HUB_ONLINE=head.
func TestHubHeadOnline(t *testing.T) {
	if os.Getenv("RAKKAN_HUB_ONLINE") != "head" {
		t.Skip("RAKKAN_HUB_ONLINE is not head")
	}
	c := newCluster(t)
	// The blob store of the test nodes is 4 GiB: enough for the two files.
	n := c.addOpts(nodeOpts{hub: &Hub{}})
	id, m, err := ImportHub(t.Context(), n.blobs.Store(), &Hub{}, "deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6", "")
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	cm, err := n.llm.coordinator(t.Context(), id, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("head of %s from the Hub in %s", m.Hub.Head, time.Since(start).Round(time.Second))
	tokens, err := cm.prompt(GenerateRequest{Prompt: "What is the capital of France?", Chat: true})
	if err != nil {
		t.Fatal(err)
	}
	in, err := cm.inputs(tokens, nil, 0, len(tokens), false)
	if err != nil {
		t.Fatal(err)
	}
	// The embedding straight to the classifier: no layers, but every
	// tensor of the head read and used.
	logits, err := cm.logits(in, len(tokens), len(tokens))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d prompt tokens, %d logits", len(tokens), len(logits))
}
