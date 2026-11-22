// Package nodesock finds a running rakkan-node's local API socket and
// speaks to it: the way applications that are not part of the node (the web
// page, the network mapper) reach it.
package nodesock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Find picks the node's API socket: -api, the "api" of -config, or
// the first that exists where the installers put it.
func Find(apiPath, configPath string) (string, error) {
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
	candidates := Candidates()
	for _, c := range candidates {
		// Windows may not mark an AF_UNIX socket as one.
		if fi, err := os.Stat(c); err == nil && (fi.Mode()&os.ModeSocket != 0 || runtime.GOOS == "windows") {
			return c, nil
		}
	}
	return "", errors.New("found no node API socket; start rakkan-node, or give -api or -config. Looked at:\n  " + strings.Join(candidates, "\n  "))
}

// Candidates are where scripts/install.sh and install.ps1 put the
// socket (user install first, then system), and the daemon's default.
func Candidates() []string {
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

// Client is an HTTP client for the node's API at sock. Use any host in URLs,
// "http://rakkan/v1/status" for example.
func Client(sock string) *http.Client {
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", sock)
		},
	}}
}
