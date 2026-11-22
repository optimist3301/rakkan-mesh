//go:build linux

// Package collect turns network activity into observations. Active scanning
// and passive watching share everything below this file: the same identifiers,
// the same observation format, the same aggregation. Only obs.Mode differs, so
// their results merge like any other observations.
package collect

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"io"
	"log"
	"net/netip"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// Session collects bodies, aggregates repeats, and signs on Flush.
type Session struct {
	Key ed25519.PrivateKey
	Pub ed25519.PublicKey
	Log *log.Logger

	mu   sync.Mutex
	aggs map[string]*agg
}

type agg struct {
	body        obs.Body
	first, last int64
	count       uint32
}

func NewSession(key ed25519.PrivateKey, logw io.Writer) *Session {
	return &Session{
		Key:  key,
		Pub:  key.Public().(ed25519.PublicKey),
		Log:  log.New(logw, "", log.Ltime),
		aggs: map[string]*agg{},
	}
}

func aggKey(b *obs.Body) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "%d|%s|", b.Mode, b.Method)
	if b.Vantage != nil {
		fmt.Fprintf(&sb, "%+v|", *b.Vantage)
	}
	sb.WriteString(strings.Join(b.Subject, ","))
	for _, c := range b.Claims {
		fmt.Fprintf(&sb, "|%s=%s@%d", c.K, c.V, c.Conf())
	}
	return sb.String()
}

// Emit records an event. Identical events (same mode, method, vantage, subject
// and claims) are folded into one observation with a time interval and count.
func (s *Session) Emit(b obs.Body) {
	b.Subject = obs.Idents(b.Subject...)
	if len(b.Subject) == 0 {
		return
	}
	if b.Time == 0 {
		b.Time = time.Now().UnixMilli()
	}
	b.Subject = slices.Clone(b.Subject)
	b.Claims = slices.DeleteFunc(slices.Clone(b.Claims), func(c obs.Claim) bool { return c.V == "" })
	b.Canonicalize()
	k := aggKey(&b)

	s.mu.Lock()
	defer s.mu.Unlock()
	a, ok := s.aggs[k]
	if !ok {
		s.aggs[k] = &agg{body: b, first: b.Time, last: b.Time, count: 1}
		return
	}
	a.count++
	a.first = min(a.first, b.Time)
	a.last = max(a.last, b.Time)
}

// Flush signs everything aggregated so far and resets.
func (s *Session) Flush() []*obs.Observation {
	s.mu.Lock()
	aggs := s.aggs
	s.aggs = map[string]*agg{}
	s.mu.Unlock()

	out := make([]*obs.Observation, 0, len(aggs))
	for _, a := range aggs {
		b := a.body
		b.Time = a.first
		if a.last > a.first {
			b.Until = a.last
		}
		if a.count > 1 {
			b.Count = a.count
		}
		o, err := obs.Sign(b, s.Key)
		if err != nil {
			s.Log.Printf("drop invalid observation %s %v: %v", b.Method, b.Subject, err)
			continue
		}
		out = append(out, o)
	}
	return out
}

func (s *Session) Pending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.aggs)
}

// KnownIPs lists the addresses discovered so far (not yet flushed), per
// network, excluding this host. Used to target follow-up probes.
func (s *Session) KnownIPs(nets []NetCtx) map[*NetCtx][]netip.Addr {
	s.mu.Lock()
	defer s.mu.Unlock()
	seen := map[string]bool{}
	out := map[*NetCtx][]netip.Addr{}
	for _, a := range s.aggs {
		for _, id := range a.body.Subject {
			if obs.Kind(id) != "ip" || seen[id] {
				continue
			}
			seen[id] = true
			ip, err := netip.ParseAddr(obs.Value(id))
			if err != nil || !ip.Is4() {
				continue
			}
			for i := range nets {
				if nets[i].NetID == obs.Scope(id) && ip != nets[i].Prefix.Addr() {
					out[&nets[i]] = append(out[&nets[i]], ip)
				}
			}
		}
	}
	return out
}

// Collector is one source of observations.
type Collector interface {
	Name() string
	Run(ctx context.Context, s *Session, nets []NetCtx) error
}

func claim(k, v string) obs.Claim            { return obs.Claim{K: k, V: v} }
func claimC(k, v string, c uint16) obs.Claim { return obs.Claim{K: k, V: v, C: c} }
