// Package web serves a web page for a node: the members it sees, the nodes
// serving language models, and a chat with a model of the network. The
// command rakkan-web runs it apart from the node, as a client of the node's
// local API (package api).
//
// The page reaches part of that API: reading the status and trust,
// describing a model, and generating. Nothing else, so it cannot run tasks
// or import models.
//
// A browser on the same machine may load pages from anywhere, and any of
// them may send requests to a port on localhost. On a loopback address the
// server therefore answers only requests naming a loopback host (against
// DNS rebinding) and takes writes only from its own origin, as JSON (which
// a foreign page cannot send without a CORS preflight, never answered). On
// any other address it also requires a token.
package web

import (
	"context"
	"crypto/subtle"
	"embed"
	"errors"
	"io/fs"
	"net"
	"net/http"
	"strings"
	"time"
)

//go:embed static
var static embed.FS

// Routes are the API routes the page may use.
var Routes = []string{
	"GET /v1/status",
	"GET /v1/trust",
	"GET /v1/llm/models/{id}",
	"POST /v1/llm/generate",
}

// Handler serves the page, and api's Routes under /v1/. With a token, API
// requests must carry it as "Authorization: Bearer <token>"; without one,
// only loopback hosts are answered.
func Handler(api http.Handler, token string) http.Handler {
	files, err := fs.Sub(static, "static")
	if err != nil {
		panic(err)
	}
	mux := http.NewServeMux()
	for _, r := range Routes {
		mux.Handle(r, api)
	}
	mux.Handle("GET /", http.FileServerFS(files))
	return &guard{next: mux, token: token}
}

type guard struct {
	next  http.Handler
	token string
}

func (g *guard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h := w.Header()
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("X-Frame-Options", "DENY")
	h.Set("Referrer-Policy", "no-referrer")
	h.Set("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
	if g.token == "" && !loopbackHost(r.Host) {
		http.Error(w, "this page answers on localhost only", http.StatusMisdirectedRequest)
		return
	}
	if strings.HasPrefix(r.URL.Path, "/v1/") {
		if g.token != "" && !g.authorized(r) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="rakkan"`)
			http.Error(w, "token required", http.StatusUnauthorized)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if o := r.Header.Get("Origin"); o != "" && o != "http://"+r.Host && o != "https://"+r.Host {
				http.Error(w, "cross-origin request", http.StatusForbidden)
				return
			}
			if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
				http.Error(w, "the body must be JSON", http.StatusUnsupportedMediaType)
				return
			}
		}
	}
	g.next.ServeHTTP(w, r)
}

func (g *guard) authorized(r *http.Request) bool {
	got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	return ok && subtle.ConstantTimeCompare([]byte(got), []byte(g.token)) == 1
}

// loopbackHost reports whether a Host header names this machine.
func loopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.TrimSuffix(strings.Trim(host, "[]"), ".")
	if strings.EqualFold(host, "localhost") || strings.HasSuffix(strings.ToLower(host), ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// IsLoopback reports whether a listen address (host:port) takes
// connections from this machine only.
func IsLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Serve serves h on l until ctx ends.
func Serve(ctx context.Context, l net.Listener, h http.Handler) error {
	srv := &http.Server{Handler: h, ReadHeaderTimeout: 10 * time.Second}
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
