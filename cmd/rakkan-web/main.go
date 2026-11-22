// Command rakkan-web serves a web page for a running rakkan-node: the
// members it sees, the nodes serving language models, and a chat with a
// model of the network.
//
// It is a client of the node like the SDKs: it talks to the node's local
// API over its unix socket (-api), and passes the page only the API routes
// it needs (package web). The node knows nothing of it. It also imports
// models from the Hugging Face Hub into the node (models.go).
//
//	rakkan-web                          # finds the socket of an installed node
//	rakkan-web -api /path/rakkan.sock   # or -config /path/node.json
//	rakkan-web -listen 0.0.0.0:7780 -token secret
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/nodesock"
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
	hub := flags.String("hub", "https://huggingface.co", "Hugging Face Hub to import models from")
	var mo mapOptions
	flags.StringVar(&mo.DB, "map-db", "", "the map store the Map tab shows (default: your own map, ~/.local/share/rakkan/map.db; with -root-key, the network's, network.db next to it). A node's own map is in <its -data>/map/map.db")
	flags.StringVar(&mo.RootKey, "root-key", "", "the network's root key: shows the WHOLE network's map, collected from the node's sealed reports (only its holder can read them); without it the Map tab shows only your own")
	flags.StringVar(&mo.CRL, "crl", "", "with -root-key: revocation list; reports of revoked members are refused")
	flags.DurationVar(&mo.Keep, "map-keep", 7*24*time.Hour, "show and keep this much history of the map; older observations are deleted from a collected network map")
	flags.DurationVar(&mo.Every, "map-collect", time.Minute, "with -root-key: how often to take new reports from the node")
	flags.StringVar(&mo.Tiles, "map-tiles", "", "basemap tiles URL template ({z}/{x}/{y}) for the geographic map; empty: none, no request leaves this machine")
	showVersion := flags.Bool("version", false, "print the version")
	flags.Parse(os.Args[1:])
	if *showVersion {
		fmt.Println("rakkan-web", version, runtime.GOOS+"/"+runtime.GOARCH, runtime.Version())
		return
	}
	if err := run(*listen, *token, *apiPath, *configPath, *hub, mo); err != nil {
		fmt.Fprintln(os.Stderr, "rakkan-web:", err)
		os.Exit(1)
	}
}

func run(listen, token, apiPath, configPath, hub string, mo mapOptions) error {
	if err := mo.validate(); err != nil {
		return err
	}
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
	pages, data, csp, err := mapHandlers(ctx, mo, sock)
	if err != nil {
		return err
	}
	app := http.NewServeMux()
	app.Handle("/app/map/", http.StripPrefix("/app/map", data))
	app.Handle("/", newImporter(hub, sock).handler())
	return web.Serve(ctx, l, web.HandlerWithMap(nodeAPI(sock), app, token, pages, csp))
}

// maxRequestBody bounds a request the page sends the node: the node's own
// bound on a generation (its prompt and images).
const maxRequestBody = 64 << 20

// nodeAPI forwards requests to the node's API socket, streaming responses
// (a generation's tokens) as they come.
//
// A request's body is read whole before it is forwarded. The proxy flushes
// the node's first tokens while its transport may still be finishing the
// request; once a response is flushed, an HTTP/1 server makes the request
// body unavailable, and the transport's late read of it failed the request,
// cutting the stream after its first line.
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
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil && r.Body != http.NoBody {
			body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBody+1))
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if len(body) > maxRequestBody {
				http.Error(w, "request too large", http.StatusRequestEntityTooLarge)
				return
			}
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
		}
		p.ServeHTTP(w, r)
	})
}

// findSocket picks the node's API socket (see nodesock.Find).
func findSocket(apiPath, configPath string) (string, error) {
	return nodesock.Find(apiPath, configPath)
}
