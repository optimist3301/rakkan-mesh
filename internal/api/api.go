// Package api serves the local HTTP API that SDKs use to submit tasks to
// the node, over a unix socket readable only by the node's user.
//
//	POST /v1/modules  body: module bytes             -> {"hash"}
//	POST /v1/run      {module|module_hash, ...}      -> result
//	POST /v1/map      {module_hash, inputs, ...}     -> NDJSON stream of results
//	GET  /v1/status                                  -> node, load and members
//
// Binary fields (module, input, output) are base64 strings in JSON. Task
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

	"github.com/optimist3301/rakkan-mesh/internal/exec"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// Server is the local API.
type Server struct {
	Service *exec.Service
	Node    *mesh.Node
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
	mux.HandleFunc("POST /v1/run", s.run)
	mux.HandleFunc("POST /v1/map", s.mapTasks)
	mux.HandleFunc("GET /v1/status", s.status)
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
	Module     []byte  `json:"module,omitempty"`
	ModuleHash string  `json:"module_hash,omitempty"`
	Entry      string  `json:"entry,omitempty"`
	Input      []byte  `json:"input,omitempty"`
	Limits     *Limits `json:"limits,omitempty"`
	Replicas   int     `json:"replicas,omitempty"`
}

type mapRequest struct {
	ModuleHash  string   `json:"module_hash"`
	Entry       string   `json:"entry,omitempty"`
	Inputs      [][]byte `json:"inputs"`
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
	Logs         string        `json:"logs,omitempty"`
	LogTruncated bool          `json:"log_truncated,omitempty"`
	GasUsed      int64         `json:"gas_used"`
	ElapsedMs    float64       `json:"elapsed_ms"`
	Node         string        `json:"node"`
	Attempts     int           `json:"attempts"`
	Replicas     []replicaJSON `json:"replicas,omitempty"`
}

type errorJSON struct {
	Index *int `json:"index,omitempty"`
	Error struct {
		Kind    string `json:"kind"`
		Message string `json:"message"`
	} `json:"error"`
}

func toJSON(res *exec.Result) resultJSON {
	out := resultJSON{
		ExitCode: res.ExitCode, Output: res.Output, Logs: string(res.Logs), LogTruncated: res.LogTruncated,
		GasUsed: res.GasUsed, ElapsedMs: float64(res.Elapsed.Microseconds()) / 1000,
		Node: res.Node.String(), Attempts: res.Attempts,
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
	res, err := s.Service.Run(r.Context(), exec.Request{
		Module: module, Entry: req.Entry, Input: req.Input, Limits: req.Limits.sandbox(), Replicas: req.Replicas,
	})
	if err != nil {
		code, e := errorOf(err)
		writeJSON(w, code, e)
		return
	}
	writeJSON(w, http.StatusOK, toJSON(res))
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
			rj := toJSON(item.Result)
			rj.Index = &idx
			enc.Encode(rj)
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	enc.Encode(map[string]any{"done": true, "reported": reported, "total": len(req.Inputs)})
}

type loadJSON struct {
	Slots  int `json:"slots"`
	Busy   int `json:"busy"`
	Queued int `json:"queued"`
}

type memberJSON struct {
	ID        string    `json:"id"`
	Roles     []string  `json:"roles"`
	Connected bool      `json:"connected"`
	Relayed   bool      `json:"relayed"`
	RTTMs     float64   `json:"rtt_ms"`
	Load      *loadJSON `json:"load,omitempty"`
}

func (s *Server) status(w http.ResponseWriter, _ *http.Request) {
	loads := s.Service.Loads()
	local := s.Service.LocalLoad()
	out := struct {
		ID      string       `json:"id"`
		Network string       `json:"network"`
		Load    loadJSON     `json:"load"`
		Members []memberJSON `json:"members"`
	}{ID: s.Node.ID().String(), Network: s.Network, Load: loadJSON{local.Slots, local.Busy, local.Queued}, Members: []memberJSON{}}
	for _, m := range s.Node.Members() {
		mj := memberJSON{ID: m.ID.String(), Roles: m.Roles, Connected: m.Connected, Relayed: m.Relayed, RTTMs: float64(m.RTT.Microseconds()) / 1000}
		if l, ok := loads[m.ID]; ok {
			mj.Load = &loadJSON{l.Slots, l.Busy, l.Queued}
		}
		out.Members = append(out.Members, mj)
	}
	writeJSON(w, http.StatusOK, out)
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
