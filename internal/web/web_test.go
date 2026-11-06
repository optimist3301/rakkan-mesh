package web

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeAPI answers 200 "api" to anything, so the tests see what reaches it.
var fakeAPI = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("api")) })

// fakeApp answers "api" too: the program's own routes are guarded alike.
var fakeApp = fakeAPI

func do(h http.Handler, method, url, host string, hdr map[string]string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, url, strings.NewReader("{}"))
	r.Host = host
	for k, v := range hdr {
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestLoopback(t *testing.T) {
	h := Handler(fakeAPI, fakeApp, "")
	json := map[string]string{"Content-Type": "application/json"}
	for _, c := range []struct {
		name, method, url, host string
		hdr                     map[string]string
		code                    int
		api                     bool
	}{
		{"page", "GET", "/", "localhost:7780", nil, 200, false},
		{"page on 127.0.0.1", "GET", "/app.js", "127.0.0.1:7780", nil, 200, false},
		{"page on ::1", "GET", "/", "[::1]:7780", nil, 200, false},
		{"rebinding", "GET", "/v1/status", "evil.example:7780", nil, http.StatusMisdirectedRequest, false},
		{"status", "GET", "/v1/status", "localhost:7780", nil, 200, true},
		{"model", "GET", "/v1/llm/models/abc", "localhost:7780", nil, 200, true},
		{"generate", "POST", "/v1/llm/generate", "localhost:7780", map[string]string{"Content-Type": "application/json", "Origin": "http://localhost:7780"}, 200, true},
		{"generate, no origin", "POST", "/v1/llm/generate", "localhost:7780", json, 200, true},
		{"foreign origin", "POST", "/v1/llm/generate", "localhost:7780", map[string]string{"Content-Type": "application/json", "Origin": "https://evil.example"}, 403, false},
		{"form post", "POST", "/v1/llm/generate", "localhost:7780", map[string]string{"Content-Type": "text/plain"}, 415, false},
		{"run is not offered", "POST", "/v1/run", "localhost:7780", json, 405, false},
		{"blobs are not offered", "GET", "/v1/blobs/x", "localhost:7780", nil, 404, false},
		{"import is not offered", "POST", "/v1/llm/models", "localhost:7780", json, 405, false},
		{"app route", "POST", "/app/models/import", "localhost:7780", json, 200, true},
		{"app route, foreign origin", "POST", "/app/models/import", "localhost:7780", map[string]string{"Content-Type": "application/json", "Origin": "https://evil.example"}, 403, false},
		{"app route, form post", "POST", "/app/models/import", "localhost:7780", map[string]string{"Content-Type": "text/plain"}, 415, false},
		{"app route, rebinding", "GET", "/app/models/jobs", "evil.example", nil, http.StatusMisdirectedRequest, false},
	} {
		w := do(h, c.method, c.url, c.host, c.hdr)
		if w.Code != c.code {
			t.Errorf("%s: %d, want %d (%s)", c.name, w.Code, c.code, strings.TrimSpace(w.Body.String()))
		}
		if got := w.Body.String() == "api"; got != c.api {
			t.Errorf("%s: reached the API: %v, want %v", c.name, got, c.api)
		}
	}
	w := do(h, "GET", "/", "localhost", nil)
	if csp := w.Header().Get("Content-Security-Policy"); !strings.Contains(csp, "default-src 'self'") {
		t.Errorf("CSP %q", csp)
	}
	if !strings.Contains(w.Body.String(), "app.js") {
		t.Error("the page is not index.html")
	}
}

func TestToken(t *testing.T) {
	h := Handler(fakeAPI, fakeApp, "s3cret")
	if w := do(h, "GET", "/", "node.lan:7780", nil); w.Code != 200 {
		t.Errorf("page without token: %d", w.Code)
	}
	if w := do(h, "GET", "/app/models/jobs", "node.lan:7780", nil); w.Code != 401 {
		t.Errorf("app route without token: %d", w.Code)
	}
	if w := do(h, "GET", "/v1/status", "node.lan:7780", nil); w.Code != 401 {
		t.Errorf("status without token: %d", w.Code)
	}
	if w := do(h, "GET", "/v1/status", "node.lan:7780", map[string]string{"Authorization": "Bearer wrong"}); w.Code != 401 {
		t.Errorf("wrong token: %d", w.Code)
	}
	if w := do(h, "GET", "/v1/status", "node.lan:7780", map[string]string{"Authorization": "Bearer s3cret"}); w.Code != 200 || w.Body.String() != "api" {
		t.Errorf("token: %d %q", w.Code, w.Body.String())
	}
}

func TestIsLoopback(t *testing.T) {
	for addr, want := range map[string]bool{
		"127.0.0.1:7780": true, "localhost:7780": true, "[::1]:7780": true,
		"0.0.0.0:7780": false, ":7780": false, "192.168.1.5:7780": false, "[::]:7780": false,
	} {
		if got := IsLoopback(addr); got != want {
			t.Errorf("IsLoopback(%q) = %v", addr, got)
		}
	}
}
