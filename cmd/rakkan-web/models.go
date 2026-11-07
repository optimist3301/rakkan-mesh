package main

// Importing models from the Hugging Face Hub into the node.
//
// rakkan-web lists a repository's files at a commit, streams each one it
// needs from the Hub into the node's blob store (POST /v1/blobs), checking
// the SHA-256 the Hub publishes for large files, and then asks the node to
// import the model from those blobs (POST /v1/llm/models). Nothing lands on
// rakkan-web's disk. A DeepSeek-V4.1 repository is handed to the node as is
// ("format": "hub"): its nodes fetch only the layers they serve.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// importer runs imports and remembers them while rakkan-web runs.
type importer struct {
	hub  string       // e.g. https://huggingface.co
	web  *http.Client // to the Hub
	node *http.Client // to the node's socket

	mu   sync.Mutex
	jobs []*job
	next int
}

func newImporter(hub, sock string) *importer {
	return &importer{
		hub: strings.TrimRight(hub, "/"),
		web: &http.Client{Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			ResponseHeaderTimeout: time.Minute,
		}},
		node: &http.Client{Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", sock)
			},
		}},
	}
}

// hfFile is a file of a repository.
type hfFile struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	SHA256 string `json:"-"` // of large (LFS) files
	done   atomic.Int64
}

// repoInfo describes a repository at a commit: what an import would fetch.
type repoInfo struct {
	Repo      string    `json:"repo"`
	Revision  string    `json:"revision"` // the commit
	ModelType string    `json:"model_type"`
	DeepSeek  bool      `json:"deepseek,omitempty"`
	Supported bool      `json:"supported"`
	Note      string    `json:"note,omitempty"`
	Files     []*hfFile `json:"files"`
	Total     int64     `json:"total"`
}

var (
	repoRe     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$`)
	revisionRe = regexp.MustCompile(`^[A-Za-z0-9._/-]{1,100}$`)
	quants     = []string{"", "bf16", "q8_0", "q4_m", "q4_0"}
	// supportedTypes are the model_type values the node imports.
	supportedTypes = []string{"llama", "qwen2", "qwen3"}
)

// hubError is an answer of the Hub that is not a success.
type hubError struct {
	status int
	msg    string
}

func (e *hubError) Error() string { return e.msg }

func (im *importer) get(ctx context.Context, u, token string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("User-Agent", "rakkan-web/"+version)
	res, err := im.web.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		res.Body.Close()
		msg := fmt.Sprintf("the Hub answered %s for %s", res.Status, u)
		switch res.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			msg = "the repository is gated or private: accept its license on the Hub and give a token"
		case http.StatusNotFound:
			msg = "no such repository, revision or file on the Hub"
		}
		return nil, &hubError{res.StatusCode, msg}
	}
	return res, nil
}

func (im *importer) getJSON(ctx context.Context, u, token string, v any) error {
	res, err := im.get(ctx, u, token)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	return json.NewDecoder(io.LimitReader(res.Body, 16<<20)).Decode(v)
}

func (im *importer) resolve(repo, rev, name string) string {
	return im.hub + "/" + repo + "/resolve/" + url.PathEscape(rev) + "/" + name
}

// inspect lists what importing repo at revision would fetch.
func (im *importer) inspect(ctx context.Context, repo, revision, token string) (*repoInfo, error) {
	if !repoRe.MatchString(repo) {
		return nil, fmt.Errorf("%q is not a repository name like Qwen/Qwen2.5-0.5B-Instruct", repo)
	}
	if revision == "" {
		revision = "main"
	}
	if !revisionRe.MatchString(revision) || strings.Contains(revision, "..") {
		return nil, fmt.Errorf("bad revision %q", revision)
	}
	var meta struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Name string `json:"rfilename"`
			Size int64  `json:"size"`
			LFS  *struct {
				SHA256 string `json:"sha256"`
				Size   int64  `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	u := im.hub + "/api/models/" + repo + "/revision/" + url.PathEscape(revision) + "?blobs=true"
	if err := im.getJSON(ctx, u, token, &meta); err != nil {
		return nil, err
	}
	if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(meta.SHA) {
		return nil, errors.New("the Hub gave no commit for the revision")
	}
	files := map[string]*hfFile{}
	for _, s := range meta.Siblings {
		f := &hfFile{Name: s.Name, Size: s.Size}
		if s.LFS != nil {
			f.SHA256, f.Size = s.LFS.SHA256, s.LFS.Size
		}
		files[s.Name] = f
	}
	info := &repoInfo{Repo: repo, Revision: meta.SHA}
	if files["config.json"] == nil {
		return nil, errors.New("the repository has no config.json: not a transformers model")
	}
	var cfg struct {
		ModelType string `json:"model_type"`
		Text      struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	if err := im.getJSON(ctx, im.resolve(repo, meta.SHA, "config.json"), token, &cfg); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	info.ModelType = cfg.ModelType
	if cfg.Text.ModelType == "deepseek_v41_text" {
		// The node imports it from the Hub itself, a layer per node.
		info.ModelType, info.DeepSeek, info.Supported = cfg.Text.ModelType, true, true
		for _, f := range files {
			if strings.HasSuffix(f.Name, ".safetensors") {
				info.Total += f.Size
			}
		}
		info.Files = []*hfFile{files["config.json"]}
		info.Note = "deepseek"
		return info, nil
	}
	info.Supported = slices.Contains(supportedTypes, info.ModelType)
	switch {
	case info.ModelType == "mistral":
		info.Note = "mistral" // its tokenizer (SentencePiece with byte fallback) is not supported yet
	case !info.Supported:
		info.Note = "unsupported"
	}

	// The weights: the shards an index names, or the safetensors files at
	// the top of the repository.
	var weights []string
	if files["model.safetensors.index.json"] != nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := im.getJSON(ctx, im.resolve(repo, meta.SHA, "model.safetensors.index.json"), token, &index); err != nil {
			return nil, fmt.Errorf("model.safetensors.index.json: %w", err)
		}
		for _, f := range index.WeightMap {
			if !slices.Contains(weights, f) {
				weights = append(weights, f)
			}
		}
	} else {
		for name := range files {
			if strings.HasSuffix(name, ".safetensors") && !strings.Contains(name, "/") {
				weights = append(weights, name)
			}
		}
	}
	slices.Sort(weights)
	if len(weights) == 0 {
		return nil, errors.New("the repository has no .safetensors weights")
	}
	if files["tokenizer.json"] == nil {
		return nil, errors.New("the repository has no tokenizer.json")
	}
	for _, name := range append([]string{"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}, weights...) {
		f := files[name]
		if f == nil {
			if slices.Contains(weights, name) {
				return nil, fmt.Errorf("the index names %s, which the repository lacks", name)
			}
			continue // the optional ones
		}
		info.Files = append(info.Files, f)
		info.Total += f.Size
	}
	return info, nil
}

// job is an import, running or ended.
type job struct {
	ID       int       `json:"id"`
	Repo     string    `json:"repo"`
	Revision string    `json:"revision"`
	Name     string    `json:"name"`
	Quant    string    `json:"quant"`
	DeepSeek bool      `json:"deepseek,omitempty"`
	Files    []*hfFile `json:"-"`
	Total    int64     `json:"total"`
	Started  time.Time `json:"started"`

	mu       sync.Mutex
	state    string // downloading, importing, done, failed, cancelled
	err      string
	model    string
	chat     bool
	finished time.Time
	cancel   context.CancelFunc
}

// jobJSON is a job as the page sees it.
type jobJSON struct {
	*job
	State    string     `json:"state"`
	Error    string     `json:"error,omitempty"`
	Model    string     `json:"model,omitempty"`
	Chat     bool       `json:"chat,omitempty"`
	Done     int64      `json:"done"`
	Finished *time.Time `json:"finished,omitempty"`
	Files    []fileJSON `json:"files"`
}

type fileJSON struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	Done int64  `json:"done"`
}

func (j *job) view() jobJSON {
	j.mu.Lock()
	defer j.mu.Unlock()
	v := jobJSON{job: j, State: j.state, Error: j.err, Model: j.model, Chat: j.chat, Files: []fileJSON{}}
	if !j.finished.IsZero() {
		t := j.finished
		v.Finished = &t
	}
	for _, f := range j.Files {
		d := f.done.Load()
		v.Files = append(v.Files, fileJSON{f.Name, f.Size, d})
		v.Done += d
	}
	return v
}

func (j *job) setState(s string) {
	j.mu.Lock()
	j.state = s
	j.mu.Unlock()
}

func (j *job) end(state, err, model string, chat bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.state, j.err, j.model, j.chat, j.finished = state, err, model, chat, time.Now()
}

// start begins importing what info describes.
func (im *importer) start(info *repoInfo, token, quant, name string) (*job, error) {
	if !slices.Contains(quants, quant) {
		return nil, fmt.Errorf("quant %q: one of bf16, q8_0, q4_m, q4_0, or none", quant)
	}
	if info.DeepSeek && quant != "" {
		return nil, errors.New("DeepSeek-V4.1 keeps its own quantization")
	}
	if !info.Supported {
		return nil, fmt.Errorf("model type %q is not supported", info.ModelType)
	}
	if name == "" {
		name = path.Base(info.Repo)
	}
	ctx, cancel := context.WithCancel(context.Background())
	im.mu.Lock()
	im.next++
	j := &job{ID: im.next, Repo: info.Repo, Revision: info.Revision, Name: name, Quant: quant, DeepSeek: info.DeepSeek,
		Files: info.Files, Total: info.Total, Started: time.Now(), state: "downloading", cancel: cancel}
	im.jobs = append(im.jobs, j)
	im.mu.Unlock()
	if info.DeepSeek {
		j.Files = nil // the node's stages fetch them
		j.setState("importing")
	}
	go im.run(ctx, j, token)
	return j, nil
}

func (im *importer) run(ctx context.Context, j *job, token string) {
	defer j.cancel()
	var req map[string]any
	if j.DeepSeek {
		req = map[string]any{"format": "hub", "repo": j.Repo, "revision": j.Revision, "name": j.Name}
	} else {
		ids := map[string]string{}
		for _, f := range j.Files {
			id, err := im.transfer(ctx, j, f, token)
			if err != nil {
				im.fail(ctx, j, fmt.Errorf("%s: %w", f.Name, err))
				return
			}
			ids[f.Name] = id
		}
		j.setState("importing")
		req = map[string]any{"format": "hf", "name": j.Name, "config": ids["config.json"], "tokenizer": ids["tokenizer.json"]}
		if j.Quant != "" {
			req["quant"] = j.Quant
		}
		for _, k := range []string{"generation_config", "tokenizer_config"} {
			if id := ids[k+".json"]; id != "" {
				req[k] = id
			}
		}
		var weights []string
		for _, f := range j.Files {
			if strings.HasSuffix(f.Name, ".safetensors") {
				weights = append(weights, ids[f.Name])
			}
		}
		req["weights"] = weights
	}
	var out struct {
		Model string `json:"model"`
		Chat  bool   `json:"chat"`
	}
	if err := im.nodeJSON(ctx, "/v1/llm/models", req, &out); err != nil {
		im.fail(ctx, j, err)
		return
	}
	j.end("done", "", out.Model, out.Chat)
}

func (im *importer) fail(ctx context.Context, j *job, err error) {
	if ctx.Err() != nil {
		j.end("cancelled", "", "", false)
		return
	}
	j.end("failed", err.Error(), "", false)
}

// transfer streams a file from the Hub into the node's blob store,
// checking its SHA-256 if the Hub published one; a broken download is
// tried again from the start, up to three times.
func (im *importer) transfer(ctx context.Context, j *job, f *hfFile, token string) (string, error) {
	var err error
	for attempt := range 3 {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(time.Duration(attempt) * 3 * time.Second):
			}
		}
		f.done.Store(0)
		var id string
		if id, err = im.transferOnce(ctx, j, f, token); err == nil {
			return id, nil
		}
		var he *hubError
		if ctx.Err() != nil || errors.As(err, &he) || errors.Is(err, errDigest) {
			return "", err
		}
	}
	return "", err
}

var errDigest = errors.New("the download does not match the Hub's SHA-256")

func (im *importer) transferOnce(ctx context.Context, j *job, f *hfFile, token string) (string, error) {
	res, err := im.get(ctx, im.resolve(j.Repo, j.Revision, f.Name), token)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	h := sha256.New()
	body := &counter{r: io.TeeReader(res.Body, h), n: &f.done}
	req, err := http.NewRequestWithContext(ctx, "POST", "http://rakkan/v1/blobs", body)
	if err != nil {
		return "", err
	}
	req.ContentLength = res.ContentLength
	var out struct {
		ID   string `json:"id"`
		Size int64  `json:"size"`
	}
	if err := im.nodeDo(req, &out); err != nil {
		return "", err
	}
	if f.SHA256 != "" && hex.EncodeToString(h.Sum(nil)) != f.SHA256 {
		return "", errDigest
	}
	if f.Size > 0 && out.Size != f.Size {
		return "", fmt.Errorf("got %d bytes of %d", out.Size, f.Size)
	}
	return out.ID, nil
}

type counter struct {
	r io.Reader
	n *atomic.Int64
}

func (c *counter) Read(p []byte) (int, error) {
	k, err := c.r.Read(p)
	c.n.Add(int64(k))
	return k, err
}

func (im *importer) nodeJSON(ctx context.Context, path string, in, out any) error {
	b, err := json.Marshal(in)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", "http://rakkan"+path, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return im.nodeDo(req, out)
}

// nodeDo sends a request to the node and decodes its JSON answer, or the
// error it gives.
func (im *importer) nodeDo(req *http.Request, out any) error {
	res, err := im.node.Do(req)
	if err != nil {
		if op := (*net.OpError)(nil); errors.As(err, &op) && op.Op == "dial" {
			return errors.New("no node at the API socket: is rakkan-node running?")
		}
		return err
	}
	defer res.Body.Close()
	data, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		var e struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(data, &e) == nil && e.Error.Message != "" {
			return errors.New("the node: " + e.Error.Message)
		}
		return fmt.Errorf("the node answered %s", res.Status)
	}
	return json.Unmarshal(data, out)
}

// Routes under /app/models/, for the page.

type importRequest struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Token    string `json:"token"`
	Quant    string `json:"quant"`
	Name     string `json:"name"`
}

func (im *importer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/models/inspect", func(w http.ResponseWriter, r *http.Request) {
		var req importRequest
		if !decode(w, r, &req) {
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), time.Minute)
		defer cancel()
		info, err := im.inspect(ctx, strings.TrimSpace(req.Repo), strings.TrimSpace(req.Revision), strings.TrimSpace(req.Token))
		if err != nil {
			fail(w, http.StatusBadRequest, err)
			return
		}
		reply(w, info)
	})
	mux.HandleFunc("POST /app/models/import", func(w http.ResponseWriter, r *http.Request) {
		var req importRequest
		if !decode(w, r, &req) {
			return
		}
		token := strings.TrimSpace(req.Token)
		ctx, cancel := context.WithTimeout(r.Context(), time.Minute)
		defer cancel()
		info, err := im.inspect(ctx, strings.TrimSpace(req.Repo), strings.TrimSpace(req.Revision), token)
		if err != nil {
			fail(w, http.StatusBadRequest, err)
			return
		}
		j, err := im.start(info, token, req.Quant, strings.TrimSpace(req.Name))
		if err != nil {
			fail(w, http.StatusBadRequest, err)
			return
		}
		reply(w, j.view())
	})
	mux.HandleFunc("GET /app/models/jobs", func(w http.ResponseWriter, r *http.Request) {
		im.mu.Lock()
		jobs := slices.Clone(im.jobs)
		im.mu.Unlock()
		out := []jobJSON{}
		for _, j := range slices.Backward(jobs) {
			out = append(out, j.view())
		}
		reply(w, out)
	})
	mux.HandleFunc("POST /app/models/jobs/{id}/cancel", func(w http.ResponseWriter, r *http.Request) {
		im.mu.Lock()
		var found *job
		for _, j := range im.jobs {
			if fmt.Sprint(j.ID) == r.PathValue("id") {
				found = j
			}
		}
		im.mu.Unlock()
		if found == nil {
			fail(w, http.StatusNotFound, errors.New("no such import"))
			return
		}
		found.cancel()
		reply(w, found.view())
	})
	return mux
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(v); err != nil {
		fail(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func reply(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, code int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"message": err.Error()}})
}
