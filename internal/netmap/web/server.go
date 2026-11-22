// Package web serves the map viewer: a single page (graph + geographic map,
// filters, time travel) and a small read-only JSON API over the resolved
// graph. Everything it needs is embedded; nothing is fetched from the
// internet except optional basemap tiles.
package web

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
)

//go:embed static
var static embed.FS

// Source provides the observation set. Version must change whenever the set
// changes (the store is append-only, so its size is enough).
type Source interface {
	Version() (string, error)
	All() ([]*obs.Observation, error)
}

type Server struct {
	Src Source
	// Tiles is a basemap URL template ({z}/{x}/{y}); empty disables the basemap.
	Tiles string
	// AllowedHosts are extra names accepted in the Host header besides
	// loopback ones (needed only when serving beyond this machine).
	AllowedHosts []string
	// Options returns the viewer's resolution options (trust) and a string
	// that changes whenever they do. Nil: trust every signer.
	Options func() (resolve.Options, string)

	mu       sync.Mutex
	ver      string
	list     []*obs.Observation
	cacheKey string
	cache    []byte
}

func (s *Server) load() ([]*obs.Observation, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, err := s.Src.Version()
	if err != nil {
		return nil, "", err
	}
	if v != s.ver || s.list == nil {
		list, err := s.Src.All()
		if err != nil {
			return nil, "", err
		}
		s.list, s.ver, s.cache, s.cacheKey = list, v, nil, ""
	}
	return s.list, s.ver, nil
}

// Handler returns the HTTP handler with security headers and Host checks.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	sub, _ := fs.Sub(static, "static")
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServerFS(sub)))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFileFS(w, r, sub, "index.html")
	})
	mux.HandleFunc("GET /api/meta", s.meta)
	mux.HandleFunc("GET /api/graph", s.graph)
	return s.guard(mux)
}

// Pages serves the viewer's page and files for embedding in another server
// (rakkan-web) under a prefix of its own: "/" is the page, "/static/" its
// files. The page reads its data from apiBase ("/app/map/", say), served by
// API; home adds a way back to the embedding page. The embedder is
// responsible for Host checks, the token and headers (see CSP).
func (s *Server) Pages(apiBase string, home bool) http.Handler {
	sub, _ := fs.Sub(static, "static")
	index, _ := fs.ReadFile(sub, "index.html")
	page := strings.Replace(string(index), `data-api="api/"`, `data-api="`+apiBase+`"`, 1)
	if home {
		page = strings.Replace(page, `data-home="0"`, `data-home="1"`, 1)
	}
	mux := http.NewServeMux()
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServerFS(sub)))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Write([]byte(page))
	})
	return mux
}

// API serves the viewer's data at /meta and /graph.
func (s *Server) API() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /meta", s.meta)
	mux.HandleFunc("GET /graph", s.graph)
	return mux
}

// CSP is the Content-Security-Policy the viewer's page needs, for an
// embedder that sets headers itself.
func (s *Server) CSP() string {
	imgSrc := "'self' data:"
	if u, err := url.Parse(strings.NewReplacer("{s}", "a", "{z}", "0", "{x}", "0", "{y}", "0").Replace(s.Tiles)); err == nil && u.Host != "" {
		imgSrc += " " + u.Scheme + "://" + u.Host
	}
	return "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src " + imgSrc +
		"; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
}

func isLoopbackHost(h string) bool {
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(strings.Trim(h, "[]"))
	return ip != nil && ip.IsLoopback()
}

// guard rejects requests whose Host is not this machine (DNS rebinding:
// otherwise any web page open in the browser could read the map through a
// hostname that resolves to 127.0.0.1) and sets restrictive headers.
func (s *Server) guard(next http.Handler) http.Handler {
	csp := s.CSP()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := r.Host
		if h, _, err := net.SplitHostPort(host); err == nil {
			host = h
		}
		if !isLoopbackHost(host) && !slices.Contains(s.AllowedHosts, host) {
			http.Error(w, "unexpected Host header", http.StatusMisdirectedRequest)
			return
		}
		h := w.Header()
		h.Set("Content-Security-Policy", csp)
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Cross-Origin-Resource-Policy", "same-origin")
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	json.NewEncoder(w).Encode(v)
}

type metaResp struct {
	Observations int      `json:"observations"`
	Observers    []string `json:"observers"`
	First        int64    `json:"first"`
	Last         int64    `json:"last"`
	Tiles        string   `json:"tiles"`
}

func (s *Server) meta(w http.ResponseWriter, r *http.Request) {
	list, _, err := s.load()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	m := metaResp{Observations: len(list), Observers: []string{}, Tiles: s.Tiles}
	seen := map[string]bool{}
	for _, o := range list {
		t1 := max(o.Time, o.Until)
		if m.First == 0 || o.Time < m.First {
			m.First = o.Time
		}
		m.Last = max(m.Last, t1)
		if k := obsKey(o); !seen[k] {
			seen[k] = true
			m.Observers = append(m.Observers, k)
		}
	}
	slices.Sort(m.Observers)
	writeJSON(w, m)
}

func obsKey(o *obs.Observation) string {
	const hex = "0123456789abcdef"
	b := make([]byte, 0, 12)
	for _, c := range o.Observer[:6] {
		b = append(b, hex[c>>4], hex[c&15])
	}
	return string(b)
}

// graph resolves the set as it was known at ?at= (unix ms): only
// observations that had started by then. Without at, the whole set.
func (s *Server) graph(w http.ResponseWriter, r *http.Request) {
	list, ver, err := s.load()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var at int64
	if v := r.URL.Query().Get("at"); v != "" {
		if at, err = strconv.ParseInt(v, 10, 64); err != nil || at <= 0 {
			http.Error(w, "bad at", http.StatusBadRequest)
			return
		}
	}
	var opts resolve.Options
	optVer := ""
	if s.Options != nil {
		opts, optVer = s.Options()
	}
	key := ver + "@" + strconv.FormatInt(at, 10) + "#" + optVer
	s.mu.Lock()
	if s.cacheKey == key {
		body := s.cache
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.Write(body)
		return
	}
	s.mu.Unlock()

	if at > 0 {
		list = slices.DeleteFunc(slices.Clone(list), func(o *obs.Observation) bool { return o.Time > at })
	}
	g := resolve.BuildWith(list, opts)
	if g.Networks == nil {
		g.Networks = []*resolve.Network{}
	}
	if g.Devices == nil {
		g.Devices = []*resolve.Device{}
	}
	if g.Links == nil {
		g.Links = []*resolve.Link{}
	}
	if g.ObserverInfo == nil {
		g.ObserverInfo = []resolve.ObserverInfo{}
	}
	body, err := json.Marshal(g)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.mu.Lock()
	s.cacheKey, s.cache = key, body
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(body)
}
