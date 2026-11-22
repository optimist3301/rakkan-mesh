package web

import (
	"bytes"
	"crypto/ed25519"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

type memSource struct {
	mu   sync.Mutex
	list []*obs.Observation
}

func (m *memSource) Version() (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return strconv.Itoa(len(m.list)), nil
}
func (m *memSource) All() ([]*obs.Observation, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*obs.Observation(nil), m.list...), nil
}

const t0 = 1767225600000

func sign(t *testing.T, dt int64, subject []string, claims ...obs.Claim) *obs.Observation {
	t.Helper()
	k := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{3}, 32))
	o, err := obs.Sign(obs.Body{Time: t0 + dt, Mode: obs.ModeActive, Method: "neigh", Subject: subject, Claims: claims}, k)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

func get(t *testing.T, h http.Handler, host, path string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodGet, path, nil)
	r.Host = host
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func setup(t *testing.T) (*memSource, http.Handler) {
	src := &memSource{list: []*obs.Observation{
		sign(t, 0, []string{"mac:00:11:32:00:00:01"}, obs.Claim{K: "name", V: "early"}),
		sign(t, 3_600_000, []string{"mac:00:11:32:00:00:02"}, obs.Claim{K: "name", V: "<img src=x onerror=alert(1)>"}),
	}}
	return src, (&Server{Src: src, Tiles: "https://tile.openstreetmap.org/{z}/{x}/{y}.png"}).Handler()
}

func TestPagesAndHeaders(t *testing.T) {
	_, h := setup(t)
	for _, p := range []string{"/", "/static/app.js", "/static/app.css", "/static/vendor/cytoscape.min.js", "/static/vendor/leaflet.js"} {
		w := get(t, h, "127.0.0.1:8742", p)
		if w.Code != 200 || w.Body.Len() == 0 {
			t.Fatalf("%s: %d", p, w.Code)
		}
		csp := w.Header().Get("Content-Security-Policy")
		if !strings.Contains(csp, "script-src 'self'") || !strings.Contains(csp, "https://tile.openstreetmap.org") ||
			strings.Contains(csp, "unsafe-eval") || w.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Fatalf("%s: headers %v", p, w.Header())
		}
	}
	if w := get(t, h, "localhost", "/static/missing.js"); w.Code != 404 {
		t.Fatalf("missing file: %d", w.Code)
	}
}

// DNS rebinding: a page on evil.example resolving to 127.0.0.1 must not be
// able to read the map.
func TestRejectsForeignHost(t *testing.T) {
	_, h := setup(t)
	for _, host := range []string{"evil.example", "evil.example:8742", "10.0.0.5:8742", "localhost.evil.example"} {
		if w := get(t, h, host, "/api/graph"); w.Code != http.StatusMisdirectedRequest {
			t.Fatalf("Host %q: %d", host, w.Code)
		}
	}
	for _, host := range []string{"localhost:8742", "127.0.0.1", "[::1]:8742"} {
		if w := get(t, h, host, "/api/meta"); w.Code != 200 {
			t.Fatalf("Host %q refused: %d", host, w.Code)
		}
	}
	allowed := (&Server{Src: &memSource{}, AllowedHosts: []string{"nas.lan"}}).Handler()
	if w := get(t, allowed, "nas.lan:8742", "/api/meta"); w.Code != 200 {
		t.Fatalf("allowed host refused: %d", w.Code)
	}
}

func TestOnlyGET(t *testing.T) {
	_, h := setup(t)
	r := httptest.NewRequest(http.MethodPost, "/api/graph", nil)
	r.Host = "localhost"
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: %d", w.Code)
	}
}

func graphAt(t *testing.T, h http.Handler, q string) *resolve.Graph {
	t.Helper()
	w := get(t, h, "localhost", "/api/graph"+q)
	if w.Code != 200 {
		t.Fatalf("graph%s: %d %s", q, w.Code, w.Body)
	}
	var g resolve.Graph
	if err := json.Unmarshal(w.Body.Bytes(), &g); err != nil {
		t.Fatal(err)
	}
	return &g
}

func TestTimeTravelAndCache(t *testing.T) {
	src, h := setup(t)
	if g := graphAt(t, h, ""); len(g.Devices) != 2 {
		t.Fatalf("now: %d devices", len(g.Devices))
	}
	if g := graphAt(t, h, "?at="+strconv.Itoa(t0+60_000)); len(g.Devices) != 1 || g.Devices[0].Label != "early" {
		t.Fatalf("at t0+1m: %+v", g.Devices)
	}
	if w := get(t, h, "localhost", "/api/graph?at=abc"); w.Code != 400 {
		t.Fatalf("bad at: %d", w.Code)
	}
	// A running watch adds observations: the next request sees them.
	src.mu.Lock()
	src.list = append(src.list, sign(t, 7_200_000, []string{"mac:00:11:32:00:00:03"}))
	src.mu.Unlock()
	if g := graphAt(t, h, ""); len(g.Devices) != 3 {
		t.Fatalf("after append: %d devices", len(g.Devices))
	}
	var m metaResp
	json.Unmarshal(get(t, h, "localhost", "/api/meta").Body.Bytes(), &m)
	if m.Observations != 3 || m.First != t0 || m.Last != t0+7_200_000 || len(m.Observers) != 1 {
		t.Fatalf("meta %+v", m)
	}
}

// Untrusted strings travel as JSON data; the API never renders HTML.
func TestUntrustedStringsStayData(t *testing.T) {
	_, h := setup(t)
	w := get(t, h, "localhost", "/api/graph")
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content type %q", ct)
	}
	if strings.Contains(w.Body.String(), "<img") {
		t.Fatal("HTML must be escaped inside JSON")
	}
}

func TestEmptyMap(t *testing.T) {
	h := (&Server{Src: &memSource{}}).Handler()
	g := graphAt(t, h, "")
	if g.Devices == nil || g.Networks == nil || g.Links == nil {
		t.Fatal("empty map must serialise as empty arrays, not null")
	}
}
