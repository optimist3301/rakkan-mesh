package exec

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"

	"github.com/optimist3301/rakkan-mesh/internal/framing"
	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// Protocol is the remote execution protocol. One task per stream:
//
//	requester -> executor  header      (JSON frame)
//	executor  -> requester offer       (JSON frame: send_module, ready, busy, rejected)
//	requester -> executor  [module]    (binary frame, only after send_module)
//	requester -> executor  input       (binary frame; empty when InputBlob is set)
//	executor  -> requester result      (JSON frame), output, logs (binary frames)
//
// Inputs and outputs larger than inlineMax travel as blobs (package blob):
// the executor fetches InputBlob from the requester and any listed
// providers, and the requester fetches OutputBlob from the executor.
//
// The requester keeps its side open until it has the result; resetting the
// stream cancels the task.
const Protocol = protocol.ID("/rakkan/exec/1.0.0")

const maxHeader = 64 << 10

type header struct {
	Version    int            `json:"v"`
	ModuleHash string         `json:"module"` // hex SHA-256
	ModuleSize int            `json:"module_size"`
	Entry      string         `json:"entry"`
	InputSize  int            `json:"input_size"`
	InputBlob  string         `json:"input_blob,omitempty"`
	Blobs      []string       `json:"blobs,omitempty"` // attached, read on demand
	Providers  []string       `json:"providers,omitempty"`
	Limits     sandbox.Limits `json:"limits"`
}

// inlineMax is the largest input or output sent within the task stream.
const inlineMax = 1 << 20

const (
	offerSendModule = "send_module"
	offerReady      = "ready"
	offerBusy       = "busy"
	offerRejected   = "rejected"
)

type offer struct {
	Status string `json:"status"`
	Reason string `json:"reason,omitempty"`
}

type resultHeader struct {
	ExitCode     uint32 `json:"exit_code"`
	GasUsed      int64  `json:"gas_used"`
	ElapsedNs    int64  `json:"elapsed_ns"`
	LogTruncated bool   `json:"log_truncated,omitempty"`
	OutputSize   int    `json:"output_size"`
	OutputBlob   string `json:"output_blob,omitempty"`
	LogsSize     int    `json:"logs_size"`
	ErrorKind    string `json:"error_kind,omitempty"`
	Error        string `json:"error,omitempty"`
}

func writeFrame(w io.Writer, b []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func readFrame(r io.Reader, limit int) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if uint64(n) > uint64(limit) {
		return nil, fmt.Errorf("frame of %d bytes exceeds %d", n, limit)
	}
	return framing.ReadFull(r, int(n))
}

func writeJSON(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return writeFrame(w, b)
}

func readJSON(r io.Reader, v any) error {
	b, err := readFrame(r, maxHeader)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// errorKinds names task errors on the wire. Names, not positions, so nodes
// running different versions still agree.
var errorKinds = map[string]error{
	"limit_policy":     sandbox.ErrLimitPolicy,
	"module_too_large": sandbox.ErrModuleTooLarge,
	"input_too_large":  sandbox.ErrInputTooLarge,
	"invalid_module":   sandbox.ErrInvalidModule,
	"forbidden_import": sandbox.ErrForbiddenImport,
	"missing_entry":    sandbox.ErrMissingEntry,
	"output_limit":     sandbox.ErrOutputLimit,
	"memory_access":    sandbox.ErrMemoryAccess,
	"out_of_gas":       sandbox.ErrOutOfGas,
	"timeout":          sandbox.ErrTimeout,
	"trap":             sandbox.ErrTrap,
	"blob_unavailable": sandbox.ErrBlobUnavailable,
	"worker_oom":       isolate.ErrWorkerOOM,
	"worker_crashed":   isolate.ErrWorkerCrashed,
	"worker_protocol":  isolate.ErrWorkerProtocol,
	"canceled":         context.Canceled,
}

// ErrorKind returns the wire name of a task error, or "internal".
func ErrorKind(err error) string {
	for name, kind := range errorKinds {
		if errors.Is(err, kind) {
			return name
		}
	}
	return "internal"
}

// RemoteError is a task error reported by an executor. It matches the
// corresponding local sentinel error with errors.Is.
type RemoteError struct {
	Kind string
	Msg  string
	Node peer.ID // the executor that reported it
}

func (e *RemoteError) Error() string { return e.Msg }

func (e *RemoteError) Unwrap() error {
	if kind, ok := errorKinds[e.Kind]; ok {
		return kind
	}
	return ErrExecutorInternal
}
