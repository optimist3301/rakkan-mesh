package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
	"github.com/optimist3301/rakkan-mesh/internal/web"
)

func TestMapTab(t *testing.T) {
	db := filepath.Join(t.TempDir(), "map.db")
	st, err := store.Open(db)
	if err != nil {
		t.Fatal(err)
	}
	_, key, _ := ed25519.GenerateKey(rand.Reader)
	o, _ := obs.Sign(obs.Body{Time: time.Now().UnixMilli(), Mode: obs.ModeActive, Method: "ping",
		Subject: []string{"ip:10.0.0.7"}, Claims: []obs.Claim{{K: "name", V: "printer"}}}, key)
	st.Add([]*obs.Observation{o})
	st.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pages, data, csp, err := mapHandlers(ctx, mapOptions{DB: db, Keep: time.Hour}, "/nonexistent.sock")
	if err != nil {
		t.Fatal(err)
	}
	app := http.NewServeMux()
	app.Handle("/app/map/", http.StripPrefix("/app/map", data))
	srv := httptest.NewServer(web.HandlerWithMap(http.NotFoundHandler(), app, "s3cret", pages, csp))
	defer srv.Close()
	get := func(path, token string) (int, string, http.Header) {
		req, _ := http.NewRequest("GET", srv.URL+path, nil)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		b, _ := io.ReadAll(res.Body)
		return res.StatusCode, string(b), res.Header
	}

	// The page and its files are open (like the main page); the data needs the token.
	code, body, h := get("/map/", "")
	if code != 200 || !strings.Contains(body, `data-api="/app/map/"`) || !strings.Contains(body, `data-home="1"`) {
		t.Fatalf("page: %d %.200s", code, body)
	}
	if !strings.Contains(h.Get("Content-Security-Policy"), "style-src 'self' 'unsafe-inline'") {
		t.Fatalf("map CSP: %q", h.Get("Content-Security-Policy"))
	}
	if code, _, _ := get("/map/static/app.js", ""); code != 200 {
		t.Fatalf("app.js: %d", code)
	}
	if code, _, h := get("/", ""); code != 200 || strings.Contains(h.Get("Content-Security-Policy"), "unsafe-inline") {
		t.Fatal("the main page must keep its strict policy")
	}
	if code, _, _ := get("/app/map/graph", ""); code != http.StatusUnauthorized {
		t.Fatalf("data without a token: %d", code)
	}
	code, body, _ = get("/app/map/graph", "s3cret")
	if code != 200 || !strings.Contains(body, "10.0.0.7") {
		t.Fatalf("graph: %d %.200s", code, body)
	}
	if code, body, _ = get("/app/map/meta", "s3cret"); code != 200 || !strings.Contains(body, `"observations":1`) {
		t.Fatalf("meta: %d %s", code, body)
	}
}
