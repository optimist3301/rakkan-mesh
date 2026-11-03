// Package api serves the local HTTP API that SDKs use to submit tasks to
// the node, over a unix socket readable only by the node's user.
//
//	POST /v1/modules     body: module bytes          -> {"hash"}
//	POST /v1/blobs       body: data, streamed        -> {"id", "size"}
//	GET  /v1/blobs/{id}                              -> data, streamed, with Range support
//	POST /v1/run         {module|module_hash, ...}   -> result
//	POST /v1/map         {module_hash, inputs, ...}  -> NDJSON stream of results
//	GET  /v1/status                                  -> node, load and members
//	POST /v1/llm/models  {checkpoint, tokenizer}     -> {"model", "config"}
//	POST /v1/llm/generate {model, prompt, ...}       -> NDJSON stream of tokens
//
// Tasks may attach up to 16 stored blobs ("blobs"), which they read in
// pieces with blob_read; a map attaches them to every task.
//
// Binary fields (module, input, output) are base64 strings in JSON. Large
// data goes through /v1/blobs instead: tasks take "input_blob" (or
// "input_blobs" for a map) and return outputs over 1 MiB as "output_blob". Task
// failures return 422 with {"error": {"kind", "message"}}; kind is one of
// exec.ErrorKind's names, or no_capacity, not_enough_nodes, no_consensus.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/exec"
	"github.com/optimist3301/rakkan-mesh/internal/llm"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	"github.com/optimist3301/rakkan-mesh/internal/trust"
)

// Server is the local API.
type Server struct {
	Service *exec.Service
	Blobs   *blob.Store
	LLM     *llm.Service // nil disables /v1/llm
	Node    *mesh.Node
	Trust   *trust.Service // nil disables /v1/trust
	Network string
}

// Listen opens the unix socket, replacing a stale one, with owner-only
// permissions.
func Listen(path string) (net.Listener, error) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	l, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		l.Close()
		return nil, err
	}
	return l, nil
}

// Handler returns the API's HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/modules", s.upload)
	mux.HandleFunc("POST /v1/blobs", s.putBlob)
	mux.HandleFunc("GET /v1/blobs/{id}", s.getBlob)
	mux.HandleFunc("POST /v1/run", s.run)
	mux.HandleFunc("POST /v1/map", s.mapTasks)
	mux.HandleFunc("GET /v1/status", s.status)
	mux.HandleFunc("GET /v1/trust", s.trust)
	mux.HandleFunc("POST /v1/llm/models", s.importModel)
	mux.HandleFunc("POST /v1/llm/generate", s.generate)
	return mux
}

// Limits are task limits as SDKs express them; zero fields take defaults.
type Limits struct {
	Gas            int64 `json:"gas,omitempty"`
	TimeoutMs      int64 `json:"timeout_ms,omitempty"`
	MemoryMB       int   `json:"memory_mb,omitempty"`
	MaxOutputBytes int   `json:"max_output_bytes,omitempty"`
	MaxInputBytes  int   `json:"max_input_bytes,omitempty"`
}

func (l *Limits) sandbox() sandbox.Limits {
	out := sandbox.DefaultLimits()
	if l == nil {
		return out
	}
	if l.Gas > 0 {
		out.MaxGas = l.Gas
	}
	if l.TimeoutMs > 0 {
		out.Timeout = time.Duration(l.TimeoutMs) * time.Millisecond
	}
	if l.MemoryMB > 0 {
		out.MaxMemoryPages = uint32(l.MemoryMB << 20 / sandbox.PageSize)
	}
	if l.MaxOutputBytes > 0 {
		out.MaxOutputBytes = l.MaxOutputBytes
	}
	if l.MaxInputBytes > 0 {
		out.MaxInputBytes = l.MaxInputBytes
	}
	return out
}

type runRequest struct {
	Module     []byte   `json:"module,omitempty"`
	ModuleHash string   `json:"module_hash,omitempty"`
	Entry      string   `json:"entry,omitempty"`
	Input      []byte   `json:"input,omitempty"`
	InputBlob  string   `json:"input_blob,omitempty"`
	Blobs      []string `json:"blobs,omitempty"`
	Limits     *Limits  `json:"limits,omitempty"`
	Replicas   int      `json:"replicas,omitempty"`
}

type mapRequest struct {
	ModuleHash  string   `json:"module_hash"`
	Entry       string   `json:"entry,omitempty"`
	Inputs      [][]byte `json:"inputs,omitempty"`
	InputBlobs  []string `json:"input_blobs,omitempty"`
	Blobs       []string `json:"blobs,omitempty"`
	Limits      *Limits  `json:"limits,omitempty"`
	Concurrency int      `json:"concurrency,omitempty"`
	// StopOnExitCode ends the map at the first task exiting with this code.
	StopOnExitCode *uint32 `json:"stop_on_exit_code,omitempty"`
}

type replicaJSON struct {
	Node   string `json:"node,omitempty"`
	Agreed bool   `json:"agreed"`
	Error  string `json:"error,omitempty"`
}

type resultJSON struct {
	Index        *int          `json:"index,omitempty"`
	ExitCode     uint32        `json:"exit_code"`
	Output       []byte        `json:"output"`
	OutputBlob   string        `json:"output_blob,omitempty"`
	OutputSize   int           `json:"output_size"`
	Logs         string        `json:"logs,omitempty"`
	LogTruncated bool          `json:"log_truncated,omitempty"`
	GasUsed      int64         `json:"gas_used"`
	ElapsedMs    float64       `json:"elapsed_ms"`
	Node         string        `json:"node"`
	Attempts     int           `json:"attempts"`
	Replicas     []replicaJSON `json:"replicas,omitempty"`
	Checked      bool          `json:"checked,omitempty"`
}

type errorJSON struct {
	Index *int `json:"index,omitempty"`
	Error struct {
		Kind    string `json:"kind"`
		Message string `json:"message"`
	} `json:"error"`
}

func (s *Server) toJSON(res *exec.Result) resultJSON {
	out := resultJSON{
		ExitCode: res.ExitCode, Output: res.Output, Logs: string(res.Logs), LogTruncated: res.LogTruncated,
		GasUsed: res.GasUsed, ElapsedMs: float64(res.Elapsed.Microseconds()) / 1000,
		Node: res.Node.String(), Attempts: res.Attempts, Checked: res.Checked,
	}
	out.OutputSize = len(res.Output)
	if !res.OutputBlob.IsZero() {
		out.OutputBlob = res.OutputBlob.String()
		if m, ok := s.Blobs.Manifest(res.OutputBlob); ok {
			out.OutputSize = int(m.Size)
		}
	}
	if out.Output == nil {
		out.Output = []byte{}
	}
	for _, r := range res.Replicas {
		rj := replicaJSON{Node: r.Node.String(), Agreed: r.Agreed}
		if r.Err != nil {
			rj.Error = r.Err.Error()
		}
		out.Replicas = append(out.Replicas, rj)
	}
	return out
}

// errorOf maps an error to a status code and wire kind.
func errorOf(err error) (int, errorJSON) {
	var e errorJSON
	e.Error.Message = err.Error()
	switch {
	case errors.Is(err, exec.ErrNoCapacity):
		e.Error.Kind = "no_capacity"
		return http.StatusServiceUnavailable, e
	case errors.Is(err, exec.ErrNotEnoughNodes):
		e.Error.Kind = "not_enough_nodes"
		return http.StatusServiceUnavailable, e
	case errors.Is(err, exec.ErrNoConsensus):
		e.Error.Kind = "no_consensus"
		return http.StatusConflict, e
	}
	e.Error.Kind = exec.ErrorKind(err)
	return http.StatusUnprocessableEntity, e
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func badRequest(w http.ResponseWriter, format string, args ...any) {
	var e errorJSON
	e.Error.Kind, e.Error.Message = "bad_request", fmt.Sprintf(format, args...)
	writeJSON(w, http.StatusBadRequest, e)
}

// maxBody bounds request bodies: a maximal module or a map's inputs,
// base64-encoded.
const maxBody = 256 << 20

func (s *Server) upload(w http.ResponseWriter, r *http.Request) {
	module, err := io.ReadAll(io.LimitReader(r.Body, int64(sandbox.DefaultLimits().MaxModuleBytes)+1))
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	if len(module) > sandbox.DefaultLimits().MaxModuleBytes {
		badRequest(w, "module larger than %d bytes", sandbox.DefaultLimits().MaxModuleBytes)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"hash": s.Service.Modules().Put(module).String()})
}

func (s *Server) putBlob(w http.ResponseWriter, r *http.Request) {
	id, m, err := s.Blobs.Put(r.Body)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id.String(), "size": m.Size})
}

func (s *Server) getBlob(w http.ResponseWriter, r *http.Request) {
	id, err := blob.ParseID(r.PathValue("id"))
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	rd, err := s.Blobs.Open(id)
	if err != nil {
		var e errorJSON
		e.Error.Kind, e.Error.Message = "not_found", err.Error()
		writeJSON(w, http.StatusNotFound, e)
		return
	}
	defer rd.Close()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("ETag", `"`+id.String()+`"`)
	http.ServeContent(w, r, "", time.Time{}, rd)
}

// blobSize returns a stored blob's size for limits, or an error for the
// client if it is unknown.
func (s *Server) blobSize(idStr string) (blob.ID, int, error) {
	id, err := blob.ParseID(idStr)
	if err != nil {
		return id, 0, err
	}
	m, ok := s.Blobs.Manifest(id)
	if !ok {
		return id, 0, fmt.Errorf("unknown blob %s: upload it first", idStr)
	}
	return id, int(m.Size), nil
}

// attached resolves attached blob IDs, which must be stored locally.
func (s *Server) attached(ids []string) ([]blob.ID, error) {
	var out []blob.ID
	for _, x := range ids {
		id, _, err := s.blobSize(x)
		if err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, nil
}

// fitInput raises the default input limit to the input's size; executors
// still enforce their own ceilings.
func fitInput(l *Limits, lim *sandbox.Limits, size int) {
	if (l == nil || l.MaxInputBytes == 0) && size > lim.MaxInputBytes {
		lim.MaxInputBytes = size
	}
}

// module resolves a request's module, inline or uploaded earlier.
func (s *Server) module(inline []byte, hash string) ([]byte, error) {
	if len(inline) > 0 {
		s.Service.Modules().Put(inline)
		return inline, nil
	}
	h, ok := exec.ParseHash(hash)
	if !ok {
		return nil, errors.New("module or a valid module_hash is required")
	}
	m, ok := s.Service.Modules().Get(h)
	if !ok {
		return nil, fmt.Errorf("unknown module %s: upload it first", hash)
	}
	return m, nil
}

func (s *Server) run(w http.ResponseWriter, r *http.Request) {
	var req runRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, maxBody)).Decode(&req); err != nil {
		badRequest(w, "%v", err)
		return
	}
	module, err := s.module(req.Module, req.ModuleHash)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	er := exec.Request{Module: module, Entry: req.Entry, Input: req.Input, Limits: req.Limits.sandbox(), Replicas: req.Replicas}
	size := len(req.Input)
	if req.InputBlob != "" {
		if er.InputBlob, size, err = s.blobSize(req.InputBlob); err != nil {
			badRequest(w, "%v", err)
			return
		}
	}
	fitInput(req.Limits, &er.Limits, size)
	if er.Blobs, err = s.attached(req.Blobs); err != nil {
		badRequest(w, "%v", err)
		return
	}
	res, err := s.Service.Run(r.Context(), er)
	if err != nil {
		code, e := errorOf(err)
		writeJSON(w, code, e)
		return
	}
	writeJSON(w, http.StatusOK, s.toJSON(res))
}

func (s *Server) mapTasks(w http.ResponseWriter, r *http.Request) {
	var req mapRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, maxBody)).Decode(&req); err != nil {
		badRequest(w, "%v", err)
		return
	}
	module, err := s.module(nil, req.ModuleHash)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	mr := exec.MapRequest{Module: module, Entry: req.Entry, Inputs: req.Inputs, Limits: req.Limits.sandbox(), Concurrency: req.Concurrency}
	for _, in := range req.Inputs {
		fitInput(req.Limits, &mr.Limits, len(in))
	}
	if len(req.Inputs) > 0 && len(req.InputBlobs) > 0 {
		badRequest(w, "give inputs or input_blobs, not both")
		return
	}
	if mr.Blobs, err = s.attached(req.Blobs); err != nil {
		badRequest(w, "%v", err)
		return
	}
	for _, x := range req.InputBlobs {
		id, size, err := s.blobSize(x)
		if err != nil {
			badRequest(w, "%v", err)
			return
		}
		mr.InputBlobs = append(mr.InputBlobs, id)
		fitInput(req.Limits, &mr.Limits, size)
	}
	if code := req.StopOnExitCode; code != nil {
		mr.StopWhen = func(it exec.MapItem) bool { return it.Err == nil && it.Result.ExitCode == *code }
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	reported := 0
	// The client disconnecting cancels the remaining tasks.
	for item := range s.Service.Map(r.Context(), mr) {
		reported++
		idx := item.Index
		if item.Err != nil {
			_, e := errorOf(item.Err)
			e.Index = &idx
			enc.Encode(e)
		} else {
			rj := s.toJSON(item.Result)
			rj.Index = &idx
			enc.Encode(rj)
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	enc.Encode(map[string]any{"done": true, "reported": reported, "total": len(req.Inputs) + len(req.InputBlobs)})
}

type loadJSON struct {
	Slots        int      `json:"slots"`
	Busy         int      `json:"busy"`
	Queued       int      `json:"queued"`
	Capabilities []string `json:"capabilities"`
}

type memberJSON struct {
	ID        string    `json:"id"`
	Roles     []string  `json:"roles"`
	Guest     bool      `json:"guest,omitempty"`
	Score     *float64  `json:"score,omitempty"`
	Connected bool      `json:"connected"`
	Relayed   bool      `json:"relayed"`
	RTTMs     float64   `json:"rtt_ms"`
	Load      *loadJSON `json:"load,omitempty"`
}

func (s *Server) status(w http.ResponseWriter, _ *http.Request) {
	loads := s.Service.Loads()
	local := s.Service.LocalLoad()
	pol := s.Node.Policy()
	out := struct {
		ID      string       `json:"id"`
		Network string       `json:"network"`
		Guest   bool         `json:"guest"`
		Open    bool         `json:"open"`
		PowBits int          `json:"pow_bits,omitempty"`
		Load    loadJSON     `json:"load"`
		Members []memberJSON `json:"members"`
		// LLM lists the nodes serving model stages, fastest first.
		LLM []llm.NodeInfo `json:"llm,omitempty"`
	}{ID: s.Node.ID().String(), Network: s.Network, Guest: !s.Node.Certified(), Open: pol.Open, PowBits: pol.PowBits,
		Load: loadJSON{local.Slots, local.Busy, local.Queued, local.Capabilities}, Members: []memberJSON{}}
	for _, m := range s.Node.Members() {
		mj := memberJSON{ID: m.ID.String(), Roles: m.Roles, Guest: m.Guest, Connected: m.Connected, Relayed: m.Relayed, RTTMs: float64(m.RTT.Microseconds()) / 1000}
		if m.Guest && s.Trust != nil {
			sc := s.Trust.Score(m.ID)
			mj.Score = &sc
		}
		if l, ok := loads[m.ID]; ok {
			mj.Load = &loadJSON{l.Slots, l.Busy, l.Queued, l.Capabilities}
		}
		out.Members = append(out.Members, mj)
	}
	if s.LLM != nil {
		out.LLM = s.LLM.Nodes()
	}
	writeJSON(w, http.StatusOK, out)
}

// trust lists the peers this node has dealt with: credit and reputation.
func (s *Server) trust(w http.ResponseWriter, _ *http.Request) {
	if s.Trust == nil {
		badRequest(w, "trust is not enabled on this node")
		return
	}
	peers := s.Trust.Peers()
	if peers == nil {
		peers = []trust.PeerInfo{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"peers": peers})
}

// importRequest names uploaded blobs: a llama2.c checkpoint and tokenizer,
// or, with format "hf", a Hugging Face model's files.
type importRequest struct {
	Format string `json:"format,omitempty"` // "llama2c" (default) or "hf"
	// Quant stores the matrices as "bf16", "q8_0", "q4_m" or "q4_0".
	Quant      string `json:"quant,omitempty"`
	Checkpoint string `json:"checkpoint,omitempty"`
	Tokenizer  string `json:"tokenizer"`
	Name       string `json:"name,omitempty"`
	// Hugging Face: config.json, the optional generation_config.json and
	// tokenizer_config.json, tokenizer.json (as Tokenizer), and the
	// safetensors files.
	Config           string   `json:"config,omitempty"`
	GenerationConfig string   `json:"generation_config,omitempty"`
	TokenizerConfig  string   `json:"tokenizer_config,omitempty"`
	Weights          []string `json:"weights,omitempty"`
}

// importModel splits a llama2.c checkpoint into a model's blobs.
func (s *Server) importModel(w http.ResponseWriter, r *http.Request) {
	if s.LLM == nil {
		badRequest(w, "language models are not enabled on this node")
		return
	}
	var req importRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		badRequest(w, "%v", err)
		return
	}
	read := func(idStr string) ([]byte, error) {
		id, err := blob.ParseID(idStr)
		if err != nil {
			return nil, err
		}
		return s.Blobs.ReadAll(id, 64<<30)
	}
	if req.Format == "hf" {
		s.importHF(w, req, read)
		return
	}
	ckData, err := read(req.Checkpoint)
	if err != nil {
		badRequest(w, "checkpoint: %v", err)
		return
	}
	tokData, err := read(req.Tokenizer)
	if err != nil {
		badRequest(w, "tokenizer: %v", err)
		return
	}
	ck, err := llm.ReadCheckpoint(ckData)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	if req.Quant != "" {
		if err := ck.Quantize(req.Quant); err != nil {
			badRequest(w, "%v", err)
			return
		}
	}
	if _, err := llm.ReadTokenizer(tokData, ck.Config.Vocab); err != nil {
		badRequest(w, "%v", err)
		return
	}
	id, m, err := llm.Import(s.Blobs, ck, tokData, req.Name)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"model": id.String(), "config": m.Config})
}

// importHF imports a Hugging Face model, reading the weights from the
// store tensor by tensor.
func (s *Server) importHF(w http.ResponseWriter, req importRequest, read func(string) ([]byte, error)) {
	src := llm.HFSource{Quant: req.Quant}
	var err error
	small := func(dst *[]byte, id, what string, required bool) bool {
		if id == "" && !required {
			return true
		}
		if *dst, err = read(id); err != nil {
			badRequest(w, "%s: %v", what, err)
			return false
		}
		return true
	}
	if !small(&src.Config, req.Config, "config", true) || !small(&src.Tokenizer, req.Tokenizer, "tokenizer", true) ||
		!small(&src.GenerationConfig, req.GenerationConfig, "generation config", false) ||
		!small(&src.TokenizerConfig, req.TokenizerConfig, "tokenizer config", false) {
		return
	}
	if len(req.Weights) == 0 {
		badRequest(w, "no weights")
		return
	}
	for _, idStr := range req.Weights {
		id, err := blob.ParseID(idStr)
		if err != nil {
			badRequest(w, "weights: %v", err)
			return
		}
		r, err := s.Blobs.Open(id)
		if err != nil {
			badRequest(w, "weights: %v", err)
			return
		}
		defer r.Close()
		src.Weights = append(src.Weights, r)
	}
	if llm.IsDeepSeek(src.Config) {
		if req.Quant != "" {
			badRequest(w, "DeepSeek-V4.1 keeps its own quantization")
			return
		}
		id, m, err := llm.ImportDeepSeek(s.Blobs, src, req.Name)
		if err != nil {
			badRequest(w, "%v", err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"model": id.String(), "config": m.ModelConfig, "chat": true})
		return
	}
	id, m, err := llm.ImportHF(s.Blobs, src, req.Name)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"model": id.String(), "config": m.Config, "chat": m.Chat != nil})
}

type generateRequest struct {
	Chat        bool    `json:"chat,omitempty"`
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	Seed        uint64  `json:"seed,omitempty"`
	Stages      int     `json:"stages,omitempty"`
}

// generate streams a generation's tokens as NDJSON, then its statistics.
func (s *Server) generate(w http.ResponseWriter, r *http.Request) {
	if s.LLM == nil {
		badRequest(w, "language models are not enabled on this node")
		return
	}
	var req generateRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		badRequest(w, "%v", err)
		return
	}
	model, err := blob.ParseID(req.Model)
	if err != nil {
		badRequest(w, "%v", err)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	stats, err := s.LLM.Generate(r.Context(), llm.GenerateRequest{
		Model: model, Prompt: req.Prompt, MaxTokens: req.MaxTokens, Temperature: req.Temperature, Seed: req.Seed, Stages: req.Stages, Chat: req.Chat,
	}, func(e llm.Event) error {
		if err := enc.Encode(map[string]any{"token": e.Token, "text": e.Text}); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	})
	if err != nil {
		var e errorJSON
		e.Error.Kind, e.Error.Message = "generation_failed", err.Error()
		if errors.Is(err, llm.ErrNoStages) {
			e.Error.Kind = "no_capacity"
		}
		enc.Encode(e)
		return
	}
	var stages []map[string]any
	for _, st := range stats.Stages {
		sj := map[string]any{"from": st.From, "to": st.To, "node": st.Node.String()}
		if st.Witness != "" {
			sj["witness"] = st.Witness.String()
		}
		stages = append(stages, sj)
	}
	enc.Encode(map[string]any{"done": true, "stats": map[string]any{
		"prompt_tokens": stats.PromptTokens, "tokens": stats.Tokens,
		"elapsed_ms":   stats.Elapsed.Milliseconds(),
		"tokens_per_s": float64(stats.Tokens) / stats.Elapsed.Seconds(),
		"stages":       stages, "failovers": stats.Failovers, "caught": stats.Caught,
	}})
}

// Serve runs the API until ctx ends.
func (s *Server) Serve(ctx context.Context, l net.Listener) error {
	srv := &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
	}()
	if err := srv.Serve(l); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
