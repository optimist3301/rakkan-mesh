package mesh

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// spool keeps sealed report packets on disk, oldest dropped first, so a
// member that was offline (the root, above all) can catch up later. Each
// packet is a file named <arrival>-<id>.rpt; the arrival number only grows,
// which makes "everything after N" a cursor.
type spool struct {
	dir      string
	maxBytes int64
	maxCount int
	ttl      time.Duration

	mu    sync.Mutex
	items []spooled // by seq
	byID  map[string]bool
	total int64
	last  int64
}

type spooled struct {
	seq  int64
	id   string
	size int64
}

// reportID names a packet: the start of its hash.
func reportID(packet []byte) string {
	h := sha256.Sum256(packet)
	return hex.EncodeToString(h[:16])
}

func openSpool(dir string, maxBytes int64, maxCount int, ttl time.Duration) (*spool, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	s := &spool{dir: dir, maxBytes: maxBytes, maxCount: maxCount, ttl: ttl, byID: map[string]bool{}}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range ents {
		name, ok := strings.CutSuffix(e.Name(), ".rpt")
		if !ok {
			if strings.HasSuffix(e.Name(), ".tmp") {
				os.Remove(filepath.Join(dir, e.Name()))
			}
			continue
		}
		seqS, id, ok := strings.Cut(name, "-")
		seq, err := strconv.ParseInt(seqS, 10, 64)
		fi, err2 := e.Info()
		if !ok || err != nil || err2 != nil || len(id) != 32 {
			continue
		}
		s.items = append(s.items, spooled{seq, id, fi.Size()})
		s.byID[id] = true
		s.total += fi.Size()
	}
	slices.SortFunc(s.items, func(a, b spooled) int { return cmp.Compare(a.seq, b.seq) })
	if n := len(s.items); n > 0 {
		s.last = s.items[n-1].seq
	}
	s.evict()
	return s, nil
}

func (s *spool) file(it spooled) string {
	return filepath.Join(s.dir, fmt.Sprintf("%020d-%s.rpt", it.seq, it.id))
}

// evict drops what is too old, then the oldest while over the limits.
// The caller holds mu.
func (s *spool) evict() {
	cut := time.Now().Add(-s.ttl).UnixNano()
	n := 0
	for n < len(s.items) && (s.items[n].seq < cut || s.total > s.maxBytes || len(s.items)-n > s.maxCount) {
		it := s.items[n]
		os.Remove(s.file(it))
		delete(s.byID, it.id)
		s.total -= it.size
		n++
	}
	s.items = s.items[n:]
}

// Put stores a packet; false means it was already there.
func (s *spool) Put(packet []byte) (id string, added bool, err error) {
	id = reportID(packet)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.byID[id] {
		return id, false, nil
	}
	if int64(len(packet)) > s.maxBytes {
		return id, false, errors.New("report larger than the spool")
	}
	seq := max(time.Now().UnixNano(), s.last+1)
	it := spooled{seq, id, int64(len(packet))}
	tmp := filepath.Join(s.dir, id+".tmp")
	if err := os.WriteFile(tmp, packet, 0o600); err != nil {
		return id, false, err
	}
	if err := os.Rename(tmp, s.file(it)); err != nil {
		os.Remove(tmp)
		return id, false, err
	}
	s.last = seq
	s.items = append(s.items, it)
	s.byID[id] = true
	s.total += it.size
	s.evict()
	return id, true, nil
}

// Has reports whether the packet with this ID is stored.
func (s *spool) Has(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.byID[id]
}

// IDs lists the newest limit stored IDs.
func (s *spool) IDs(limit int) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	from := max(0, len(s.items)-limit)
	out := make([]string, 0, len(s.items)-from)
	for _, it := range s.items[from:] {
		out = append(out, it.id)
	}
	return out
}

// Packet is a stored report with its place in arrival order.
type Packet struct {
	Seq  int64
	ID   string
	Data []byte
}

// After returns up to limit packets that arrived after seq (0: from the
// start), oldest first, and no more than maxBytes of them (one at least).
func (s *spool) After(seq int64, limit int, maxBytes int64, skip map[string]bool) []Packet {
	s.mu.Lock()
	i, _ := slices.BinarySearchFunc(s.items, seq, func(it spooled, seq int64) int { return cmp.Compare(it.seq, seq) })
	for i < len(s.items) && s.items[i].seq <= seq {
		i++
	}
	var list []spooled
	var size int64
	for _, it := range s.items[i:] {
		if skip[it.id] {
			continue
		}
		if len(list) >= limit || (len(list) > 0 && size+it.size > maxBytes) {
			break
		}
		list = append(list, it)
		size += it.size
	}
	s.mu.Unlock()
	var out []Packet
	for _, it := range list {
		b, err := os.ReadFile(s.file(it))
		if err != nil {
			continue // evicted meanwhile
		}
		out = append(out, Packet{it.seq, it.id, b})
	}
	return out
}
