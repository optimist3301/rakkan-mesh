package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/collector"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
	netmapweb "github.com/optimist3301/rakkan-mesh/internal/netmap/web"
	"github.com/optimist3301/rakkan-mesh/internal/nodesock"
)

// mapOptions say what map the page's "Map" shows.
//
// Without RootKey it is this installation's own map: what its node scanned
// and kept (-map-db). With the root key it is the whole network: rakkan-web
// takes the sealed reports from the node, opens them with the key (it is read
// here and goes nowhere else) into the network store and shows that. Without
// the key nobody can see more than their own map: the reports are sealed to it.
type mapOptions struct {
	DB      string
	RootKey string
	CRL     string
	Keep    time.Duration
	Every   time.Duration
	Tiles   string
}

// mapHandlers returns the viewer's page, its data (for /app/map/) and the
// policy of its page. It starts the collection when there is a root key.
func mapHandlers(ctx context.Context, o mapOptions, sock string) (pages, data http.Handler, csp string, err error) {
	db := o.DB
	if o.RootKey != "" {
		if db == "" {
			db = collector.DefaultDB()
		}
		root, err := collector.LoadRoot(o.RootKey, o.CRL)
		if err != nil {
			return nil, nil, "", err
		}
		c := &collector.Collector{Root: root, Client: nodesock.Client(sock), DB: db, Keep: o.Keep}
		go func() {
			for {
				stats, err := c.Once()
				switch {
				case err != nil:
					slog.Warn("collecting the network map", "err", err)
				case stats.Packets > 0 || stats.Pruned > 0:
					slog.Info("collected the network map", "reports", stats.Packets, "new", stats.Added, "refused", stats.Bad, "dropped", stats.Pruned)
				}
				select {
				case <-ctx.Done():
					return
				case <-time.After(o.Every):
				}
			}
		}()
	} else if db == "" {
		db = store.DefaultPath()
	}
	if o.RootKey == "" {
		if _, err := os.Stat(db); errors.Is(err, os.ErrNotExist) {
			slog.Info("no map yet at " + db + ": the page shows an empty one")
		}
	}
	src := netmapweb.StoreSource{Path: db, Keep: o.Keep}
	srv := &netmapweb.Server{Src: src, Tiles: o.Tiles, Options: src.Options}
	slog.Info("map", "store", db, "whole_network", o.RootKey != "")
	return srv.Pages("/app/map/", true), srv.API(), srv.CSP(), nil
}

func (o mapOptions) validate() error {
	if o.RootKey == "" && o.CRL != "" {
		return fmt.Errorf("-crl is for -root-key")
	}
	return nil
}
