//go:build !linux

// Package cli is the network map: `rakkan-node map`. It collects with Linux
// interfaces (netlink, raw sockets, NetworkManager), so it exists on Linux only.
package cli

import (
	"context"
	"fmt"
	"os"
	"time"
)

// Main reports that the map is not available here.
func Main([]string) int {
	fmt.Fprintln(os.Stderr, "rakkan-node map runs on Linux only")
	return 1
}

// Reporter is unavailable off Linux.
type Reporter struct{}

// Run does nothing.
func (Reporter) Run(context.Context) {}

// ReporterOptions are unused off Linux.
type ReporterOptions struct {
	DB, Key       string
	Every         time.Duration
	Scan          bool
	Keep          time.Duration
	Location, WAN string
	Send          func([]byte) error
}

// NewReporter returns an error off Linux.
func NewReporter(ReporterOptions) (*Reporter, error) {
	return nil, fmt.Errorf("the network map runs on Linux only")
}
