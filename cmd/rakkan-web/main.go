// Command rakkan-web serves a web page for a running rakkan-node: the
// members it sees, the nodes serving language models, and a chat with a
// model of the network.
//
// It is a client of the node like the SDKs: it talks to the node's local
// API over its unix socket (-api), and passes the page only the API routes
// it needs (package web). The node knows nothing of it.
//
//	rakkan-web                          # finds the socket of an installed node
//	rakkan-web -api /path/rakkan.sock   # or -config /path/node.json
//	rakkan-web -listen 0.0.0.0:7780 -token secret
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/web"
)

// version is set at release build time: -ldflags "-X main.version=...".
var version = "dev"

func main() {
	flags := flag.NewFlagSet("rakkan-web", flag.ExitOnError)
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "usage: rakkan-web [flags]\n\nA web page for a running rakkan-node: members, models and a chat.\n\n")
		flags.PrintDefaults()
	}
	listen := flags.String("listen", "127.0.0.1:7780", "address of the page")
	token := flags.String("token", os.Getenv("RAKKAN_WEB_TOKEN"), "token the page asks for; required when -listen is not a loopback address (default $RAKKAN_WEB_TOKEN)")
	apiPath := flags.String("api", "", "the node's API socket (default: from -config, or where the installer puts it)")
	configPath := flags.String("config", "", "the node's JSON config, to take the socket from its \"api\" field")
	showVersion := flags.Bool("version", false, "print the version")
	flags.Parse(os.Args[1:])
	if *showVersion {
		fmt.Println("rakkan-web", version, runtime.GOOS+"/"+runtime.GOARCH, runtime.Version())
		return
	}
	if err := run(*listen, *token, *apiPath, *configPath); err != nil {
		fmt.Fprintln(os.Stderr, "rakkan-web:", err)
		os.Exit(1)
	}
}

func run(listen, token, apiPath, configPath string) error {
	if !web.IsLoopback(listen) && token == "" {
		return fmt.Errorf("-listen %s is reachable from other machines: set -token (or RAKKAN_WEB_TOKEN), or use 127.0.0.1", listen)
	}
	sock, err := findSocket(apiPath, configPath)
	if err != nil {
		return err
	}
	l, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	host := l.Addr().String()
	if web.IsLoopback(listen) {
		host = strings.Replace(host, "127.0.0.1", "localhost", 1)
	}
	fmt.Printf("rakkan-web %s: http://%s, node API at %s\n", version, host, sock)
	return web.Serve(ctx, l, web.Handler(nodeAPI(sock), token))
}

// nodeAPI forwards requests to the node's API socket, streaming responses
// (a generation's tokens) as they come.
func nodeAPI(sock string) http.Handler {
	p := &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(&url.URL{Scheme: "http", Host: "rakkan"})
			r.Out.Header.Del("Authorization") // the page's token is not the node's business
			r.Out.Header.Del("Cookie")
		},
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", sock)
			},
			ResponseHeaderTimeout: 10 * time.Minute, // a model's first token may take a while
		},
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Warn("node API", "path", r.URL.Path, "err", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			kind, msg := "node_failed", "the node broke off the request: see its log"
			if op := (*net.OpError)(nil); errors.As(err, &op) && op.Op == "dial" {
				kind, msg = "node_unreachable", "no node at "+sock+": is rakkan-node running?"
			}
			json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"kind": kind, "message": msg}})
		},
	}
	return p
}

// findSocket picks the node's API socket: -api, the "api" of -config, or
// the first that exists where the installers put it.
func findSocket(apiPath, configPath string) (string, error) {
	if apiPath != "" {
		return apiPath, nil
	}
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return "", err
		}
		var cf struct {
			API string `json:"api"`
		}
		if err := json.Unmarshal(data, &cf); err != nil {
			return "", fmt.Errorf("%s: %w", configPath, err)
		}
		if cf.API == "" {
			return "", fmt.Errorf("%s has no \"api\" socket", configPath)
		}
		if !filepath.IsAbs(cf.API) {
			cf.API = filepath.Join(filepath.Dir(configPath), cf.API)
		}
		return cf.API, nil
	}
	candidates := socketCandidates()
	for _, c := range candidates {
		// Windows may not mark an AF_UNIX socket as one.
		if fi, err := os.Stat(c); err == nil && (fi.Mode()&os.ModeSocket != 0 || runtime.GOOS == "windows") {
			return c, nil
		}
	}
	return "", errors.New("found no node API socket; start rakkan-node, or give -api or -config. Looked at:\n  " + strings.Join(candidates, "\n  "))
}

// socketCandidates are where scripts/install.sh and install.ps1 put the
// socket (user install first, then system), and the daemon's default.
func socketCandidates() []string {
	home, _ := os.UserHomeDir()
	var out []string
	switch runtime.GOOS {
	case "windows":
		if d := os.Getenv("LOCALAPPDATA"); d != "" {
			out = append(out, filepath.Join(d, "Rakkan", "data", "rakkan.sock"))
		}
		if d := os.Getenv("ProgramData"); d != "" {
			out = append(out, filepath.Join(d, "Rakkan", "data", "rakkan.sock"))
		}
	case "darwin":
		out = append(out, filepath.Join(home, "Library", "Application Support", "Rakkan", "data", "rakkan.sock"),
			"/Library/Application Support/Rakkan/data/rakkan.sock")
	default:
		data := os.Getenv("XDG_DATA_HOME")
		if data == "" {
			data = filepath.Join(home, ".local", "share")
		}
		out = append(out, filepath.Join(data, "rakkan", "rakkan.sock"), "/var/lib/rakkan/rakkan.sock")
	}
	return append(out, "rakkan.sock")
}
