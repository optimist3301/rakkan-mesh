package llm

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
)

// A model on the Hugging Face Hub, as large as DeepSeek-V4.1-Flash (475
// GiB in 48 files), is not imported through one node. Its manifest pins
// the repository's revision and each file's size and hash (the LFS SHA-256
// of weight files, the git blob SHA-1 of small ones), and says which files
// hold which layer. A node serving layers gets just their files: from a
// member that has them, else from the Hub, checking the hash as the bytes
// arrive, and reads the tensors from them in place (sources.go). The
// import downloads only the small files: config.json, the index and the
// tokenizer.

// Hub reads files of Hugging Face repositories.
type Hub struct {
	// Endpoint is the Hub's address, https://huggingface.co by default.
	Endpoint string
	// Token, if set, authorizes downloads of gated repositories.
	Token  string
	Client *http.Client
}

func (h *Hub) endpoint() string {
	if h.Endpoint == "" {
		return "https://huggingface.co"
	}
	return strings.TrimSuffix(h.Endpoint, "/")
}

func (h *Hub) client() *http.Client {
	if h.Client != nil {
		return h.Client
	}
	return http.DefaultClient
}

func (h *Hub) get(ctx context.Context, u string, header map[string]string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	for k, v := range header {
		req.Header.Set(k, v)
	}
	return h.client().Do(req)
}

// HubFile is a file of a repository at a revision, as the manifest pins
// it: its size and its LFS SHA-256, or for a file outside LFS its git blob
// SHA-1.
type HubFile struct {
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
	SHA1   string `json:"sha1,omitempty"`
}

// key names a file by its content.
func (f HubFile) key() string {
	if f.SHA256 != "" {
		return "sha256:" + f.SHA256
	}
	return "sha1:" + f.SHA1
}

// hasher hashes a file's bytes as its pin says.
func (f HubFile) hasher() hash.Hash {
	if f.SHA256 != "" {
		return sha256.New()
	}
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", f.Size)
	return h
}

func (f HubFile) check(h hash.Hash) bool {
	want := f.SHA256
	if want == "" {
		want = f.SHA1
	}
	return hex.EncodeToString(h.Sum(nil)) == want
}

var hexHash = regexp.MustCompile(`^[0-9a-f]{40}([0-9a-f]{24})?$`)

func (f HubFile) valid() bool {
	return f.Size >= 0 && (len(f.SHA256) == 64 && hexHash.MatchString(f.SHA256) || f.SHA256 == "" && len(f.SHA1) == 40 && hexHash.MatchString(f.SHA1))
}

// Tree lists a repository's files at a revision, by path.
func (h *Hub) Tree(ctx context.Context, repo, revision string) (map[string]HubFile, error) {
	u := fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=true", h.endpoint(), repo, url.PathEscape(revision))
	files := map[string]HubFile{}
	for u != "" {
		resp, err := h.get(ctx, u, nil)
		if err != nil {
			return nil, err
		}
		var entries []struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
			OID  string `json:"oid"`
			LFS  *struct {
				OID  string `json:"oid"`
				Size int64  `json:"size"`
			} `json:"lfs"`
		}
		err = func() error {
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("hub: %s: %s", u, resp.Status)
			}
			return json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&entries)
		}()
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if e.Type != "file" {
				continue
			}
			f := HubFile{Size: e.Size, SHA1: e.OID}
			if e.LFS != nil {
				f = HubFile{Size: e.LFS.Size, SHA256: e.LFS.OID}
			}
			files[e.Path] = f
		}
		u = nextLink(resp.Header.Get("Link"))
	}
	return files, nil
}

// nextLink is the rel="next" URL of a Link header, if any.
func nextLink(link string) string {
	for _, part := range strings.Split(link, ",") {
		if strings.Contains(part, `rel="next"`) {
			if a, b := strings.Index(part, "<"), strings.Index(part, ">"); a >= 0 && b > a {
				return part[a+1 : b]
			}
		}
	}
	return ""
}

// open streams a file of a repository, resuming where a broken download
// stopped (with a Range request), and checks its size and hash: the
// reader returns an error at the end if they are not the pinned ones.
func (h *Hub) open(ctx context.Context, repo, revision, path string, f HubFile) io.Reader {
	return &hubReader{h: h, ctx: ctx, url: fmt.Sprintf("%s/%s/resolve/%s/%s", h.endpoint(), repo, url.PathEscape(revision), path),
		f: f, sum: f.hasher()}
}

type hubReader struct {
	h     *Hub
	ctx   context.Context
	url   string
	f     HubFile
	sum   hash.Hash
	off   int64
	body  io.ReadCloser
	tries int // failures in a row
	// verified: all of the file was read and is the pinned one.
	verified bool
}

func (r *hubReader) Read(p []byte) (int, error) {
	for {
		if r.off == r.f.Size {
			if r.body != nil {
				r.body.Close()
				r.body = nil
			}
			if !r.f.check(r.sum) {
				return 0, fmt.Errorf("hub: %s does not match its pinned hash", r.url)
			}
			r.verified = true
			return 0, io.EOF
		}
		if r.body == nil {
			if err := r.connect(); err != nil {
				return 0, err
			}
		}
		p = p[:min(int64(len(p)), r.f.Size-r.off)]
		n, err := r.body.Read(p)
		if n > 0 {
			r.sum.Write(p[:n])
			r.off += int64(n)
			r.tries = 0
			return n, nil
		}
		// A connection lost: again from where it stopped.
		r.body.Close()
		r.body = nil
		if err == nil || err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		if r.tries++; r.tries > 5 || r.ctx.Err() != nil {
			// Not wrapped: a reader taking io.ErrUnexpectedEOF for the end
			// would keep the file cut short.
			return 0, fmt.Errorf("hub: %s at byte %d: %v", r.url, r.off, err)
		}
		select {
		case <-time.After(time.Duration(r.tries) * time.Second):
		case <-r.ctx.Done():
			return 0, r.ctx.Err()
		}
	}
}

func (r *hubReader) connect() error {
	var header map[string]string
	if r.off > 0 {
		header = map[string]string{"Range": "bytes=" + strconv.FormatInt(r.off, 10) + "-"}
	}
	resp, err := r.h.get(r.ctx, r.url, header)
	if err != nil {
		return err
	}
	switch {
	case r.off > 0 && resp.StatusCode == http.StatusPartialContent:
	case r.off == 0 && resp.StatusCode == http.StatusOK:
	default:
		resp.Body.Close()
		return fmt.Errorf("hub: %s: %s", r.url, resp.Status)
	}
	r.body = resp.Body
	return nil
}

// fetchSmall downloads a small file whole, checked.
func (h *Hub) fetchSmall(ctx context.Context, repo, revision, path string, f HubFile, max int64) ([]byte, error) {
	if f.Size > max {
		return nil, fmt.Errorf("hub: %s is %d bytes, more than %d", path, f.Size, max)
	}
	r := h.open(ctx, repo, revision, path, f)
	b, err := io.ReadAll(r)
	if err == nil && !r.(*hubReader).verified {
		err = fmt.Errorf("hub: %s incomplete", path)
	}
	return b, err
}

// HubOrigin is where a model's weights come from: files of a Hub
// repository at a revision, and which of them hold the coordinator's part
// (embedding, final norm, classifier) and each layer.
type HubOrigin struct {
	Repo     string             `json:"repo"`
	Revision string             `json:"revision"`
	Files    map[string]HubFile `json:"files"`
	Head     []string           `json:"head"`
	Layers   [][]string         `json:"layers"`
}

var commitID = regexp.MustCompile(`^[0-9a-f]{40}$`)

func (o *HubOrigin) check(c *dsv41.Config) error {
	switch {
	case o.Repo == "" || strings.Contains(o.Repo, "..") || strings.Count(o.Repo, "/") != 1:
		return fmt.Errorf("repository %q", o.Repo)
	case !commitID.MatchString(o.Revision):
		return fmt.Errorf("revision %q is not a commit", o.Revision)
	case len(o.Layers) != c.Layers:
		return fmt.Errorf("%d layers' files, the config says %d", len(o.Layers), c.Layers)
	}
	for name, f := range o.Files {
		if !f.valid() || strings.Contains(name, "..") {
			return fmt.Errorf("file %q pinned as %+v", name, f)
		}
	}
	for _, names := range append([][]string{o.Head, {"tokenizer.json"}}, o.Layers...) {
		if len(names) == 0 {
			return errors.New("a part without files")
		}
		for _, n := range names {
			if _, ok := o.Files[n]; !ok {
				return fmt.Errorf("file %q not pinned", n)
			}
		}
	}
	return nil
}

// ImportHub stores the manifest of a DeepSeek-V4.1 model on the Hub,
// pinned at a revision (a branch or tag is resolved to its commit), and
// returns its ID. Only config.json, the index and the tokenizer are
// downloaded; the weights are fetched by the nodes that serve them.
func ImportHub(ctx context.Context, store *blob.Store, hub *Hub, repo, revision, name string) (blob.ID, *Manifest, error) {
	if revision == "" {
		revision = "main"
	}
	if !commitID.MatchString(revision) {
		var err error
		if revision, err = hub.commit(ctx, repo, revision); err != nil {
			return blob.ID{}, nil, err
		}
	}
	files, err := hub.Tree(ctx, repo, revision)
	if err != nil {
		return blob.ID{}, nil, err
	}
	small := func(path string) ([]byte, error) {
		f, ok := files[path]
		if !ok {
			return nil, fmt.Errorf("hub: %s has no %s", repo, path)
		}
		return hub.fetchSmall(ctx, repo, revision, path, f, 64<<20)
	}
	cfgData, err := small("config.json")
	if err != nil {
		return blob.ID{}, nil, err
	}
	if !IsDeepSeek(cfgData) {
		return blob.ID{}, nil, errors.New("hub: only DeepSeek-V4.1 models are imported from the Hub")
	}
	c, err := dsv41.ParseConfig(cfgData)
	if err != nil {
		return blob.ID{}, nil, err
	}
	tok, err := small("tokenizer.json")
	if err != nil {
		return blob.ID{}, nil, err
	}
	bpe, err := ReadBPE(tok, c.Vocab)
	if err != nil {
		return blob.ID{}, nil, err
	}
	if len(c.EngramLayers) > 0 {
		// The Engram hash tables come from the tokenizer: it must be the
		// model's.
		if _, err := dsv41.BuildEngramTables(c, bpe); err != nil {
			return blob.ID{}, nil, err
		}
	}
	index, err := small("model.safetensors.index.json")
	if err != nil {
		return blob.ID{}, nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(index, &idx); err != nil {
		return blob.ID{}, nil, fmt.Errorf("hub: index: %w", err)
	}
	o := &HubOrigin{Repo: repo, Revision: revision, Files: map[string]HubFile{}, Layers: make([][]string, c.Layers)}
	add := func(dst *[]string, file string) {
		if !slices.Contains(*dst, file) {
			*dst = append(*dst, file)
		}
	}
	for tensor, file := range idx.WeightMap {
		if _, ok := files[file]; !ok {
			return blob.ID{}, nil, fmt.Errorf("hub: the index names %s, not in the repository", file)
		}
		if l, ok := dsv41.TensorLayer(tensor); ok && l < c.Layers {
			add(&o.Layers[l], file)
		} else if dsv41.IsHeadTensor(tensor) {
			add(&o.Head, file)
		}
	}
	for _, names := range append([][]string{o.Head}, o.Layers...) {
		sort.Strings(names)
		for _, n := range names {
			o.Files[n] = files[n]
		}
	}
	o.Files["tokenizer.json"] = files["tokenizer.json"]
	m := &Manifest{Version: 2, Name: name, Arch: "dsv41", ModelConfig: cfgData, TokenizerFormat: "hf", Hub: o}
	data, _ := json.Marshal(m)
	parsed, err := ParseManifest(data)
	if err != nil {
		return blob.ID{}, nil, err
	}
	id, err := store.PutBytes(data)
	return id, parsed, err
}

// commit resolves a branch or tag to its commit.
func (h *Hub) commit(ctx context.Context, repo, revision string) (string, error) {
	resp, err := h.get(ctx, fmt.Sprintf("%s/api/models/%s/revision/%s", h.endpoint(), repo, url.PathEscape(revision)), nil)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("hub: %s at %s: %s", repo, revision, resp.Status)
	}
	var info struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&info); err != nil {
		return "", err
	}
	if !commitID.MatchString(info.SHA) {
		return "", fmt.Errorf("hub: %s at %s: commit %q", repo, revision, info.SHA)
	}
	return info.SHA, nil
}
