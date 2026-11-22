package web

import (
	"crypto/sha256"
	"fmt"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
)

// StoreSource shows a store on disk without holding it open: each call opens
// it, reads and closes it, so the process that writes to it (a collector, a
// scan) is never blocked, and nothing is held between requests but the
// resolved set the viewer caches.
//
// Only the last Keep of observations are shown, at most Limit of them (0:
// everything): a store the node fills for months must not become a viewer
// that needs gigabytes.
type StoreSource struct {
	Path  string
	Keep  time.Duration
	Limit int
}

// DefaultViewLimit bounds the observations a viewer resolves at once.
const DefaultViewLimit = 200_000

func (s StoreSource) open() (*store.Store, error) { return store.Open(s.Path) }

// Version implements Source.
func (s StoreSource) Version() (string, error) {
	st, err := s.open()
	if err != nil {
		return "", err
	}
	defer st.Close()
	return st.Version()
}

// All implements Source.
func (s StoreSource) All() ([]*obs.Observation, error) {
	st, err := s.open()
	if err != nil {
		return nil, err
	}
	defer st.Close()
	var cutoff int64
	if s.Keep > 0 {
		cutoff = time.Now().Add(-s.Keep).UnixMilli()
	}
	limit := s.Limit
	if limit == 0 {
		limit = DefaultViewLimit
	}
	return st.Since(cutoff, limit)
}

// Options returns the viewer's options for the store: its trust book, re-read when it changes (no AS table).
func (s StoreSource) Options() (resolve.Options, string) {
	st, err := s.open()
	if err != nil {
		return resolve.Options{}, "err"
	}
	defer st.Close()
	book, err := st.TrustBook()
	if err != nil {
		return resolve.Options{}, "err"
	}
	o := resolve.Options{Trust: book}
	v := ""
	for _, e := range book.Entries() {
		v += e.Key + e.Level.String()
	}
	return o, fmt.Sprintf("%x", sha256.Sum256([]byte(v)))
}
