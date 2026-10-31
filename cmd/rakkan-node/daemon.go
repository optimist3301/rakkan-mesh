package main

import (
	"context"
	"runtime"

	"errors"
	"flag"
	"fmt"
	"github.com/optimist3301/rakkan-mesh/internal/api"
	"github.com/optimist3301/rakkan-mesh/internal/exec"
	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	"io/fs"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
)

func daemonCmd(args []string) int {
	flags := flag.NewFlagSet("daemon", flag.ExitOnError)
	network := flags.String("network", "", "network ID (required)")
	keyPath := flags.String("key", "node.key", "node private key")
	certPath := flags.String("cert", "node.cert", "node certificate")
	crlPath := flags.String("crl", "", "revocation list; reloaded and published to the network when it changes")
	listen := flags.String("listen", "", "comma-separated listen multiaddrs (default: QUIC and TCP on all interfaces)")
	bootstrap := flags.String("bootstrap", "", "comma-separated multiaddrs of known members, ending in /p2p/<peer ID>")
	peerCache := flags.String("peers", "peers.json", "file remembering members' addresses across restarts; empty disables")
	mdnsOn := flags.Bool("mdns", true, "discover members on the local network")
	nat := flags.Bool("nat", true, "UPnP port mapping, hole punching and AutoNAT")
	relay := flags.Bool("relay", false, "relay traffic for members; enable on publicly reachable nodes")
	reach := flags.String("reachability", "auto", "auto (from interface addresses), public, or private (behind NAT)")
	verbose := flags.Bool("v", false, "debug logging")
	workers := flags.Int("workers", runtime.GOMAXPROCS(0), "tasks this node runs for the network at once; 0 only submits tasks")
	isolation := flags.String("isolation", "strict", "worker isolation: strict, ns or process")
	cgroup := flags.String("cgroup", "", `delegated cgroup v2 directory for per-worker limits, or "self"`)
	apiPath := flags.String("api", "rakkan.sock", "unix socket for the local API used by SDKs; empty disables")
	statusEvery := flags.Duration("status", 30*time.Second, "how often to print the member table")
	flags.Parse(args)

	n, err := membership.ParseNetwork(*network)
	if err != nil {
		return fail(fmt.Errorf("-network: %w", err))
	}
	key, err := membership.LoadKey(*keyPath)
	if err != nil {
		return fail(err)
	}
	cert, err := membership.LoadCert(*certPath)
	if err != nil {
		return fail(err)
	}
	reachability, err := mesh.ParseReachability(*reach)
	if err != nil {
		return fail(err)
	}
	cfg := mesh.Config{Key: key, Network: n, Cert: cert, MDNS: *mdnsOn, NAT: *nat, RelayService: *relay, Reachability: reachability, PeerCache: *peerCache}
	if *crlPath != "" {
		// A missing file is fine: it is picked up once the operator creates it.
		if cfg.CRL, err = membership.LoadCRL(*crlPath); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return fail(err)
		}
	}
	if *listen != "" {
		cfg.ListenAddrs = strings.Split(*listen, ",")
	}
	for _, s := range strings.Split(*bootstrap, ",") {
		if s == "" {
			continue
		}
		ai, err := peer.AddrInfoFromString(s)
		if err != nil {
			return fail(fmt.Errorf("-bootstrap %s: %w", s, err))
		}
		cfg.Bootstrap = append(cfg.Bootstrap, *ai)
	}
	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	node, err := mesh.New(cfg)
	if err != nil {
		return fail(err)
	}
	defer node.Close()
	for _, a := range node.AddrInfo().Addrs {
		fmt.Printf("listening on %s/p2p/%s\n", a, node.ID())
	}

	ceiling := sandbox.DefaultLimits()
	svcCfg := exec.Config{Node: node, Ceiling: ceiling, Logger: cfg.Logger}
	if *workers > 0 {
		iso, err := parseIsolation(*isolation)
		if err != nil {
			return fail(err)
		}
		iso.Cgroup = *cgroup
		pool, err := isolate.NewPool(isolate.Config{Ceiling: ceiling, Workers: *workers, Isolation: iso})
		if err != nil {
			return fail(err)
		}
		defer pool.Close()
		svcCfg.Pool, svcCfg.Slots = pool, *workers
	}
	svc, err := exec.New(svcCfg)
	if err != nil {
		return fail(err)
	}
	defer svc.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *apiPath != "" {
		l, err := api.Listen(*apiPath)
		if err != nil {
			return fail(fmt.Errorf("api: %w", err))
		}
		defer os.Remove(*apiPath)
		srv := &api.Server{Service: svc, Node: node, Network: n.ID}
		go func() {
			if err := srv.Serve(ctx, l); err != nil {
				slog.Error("api", "err", err)
			}
		}()
		fmt.Printf("api on %s\n", *apiPath)
	}
	status := time.NewTicker(*statusEvery)
	defer status.Stop()
	reload := time.NewTicker(5 * time.Second)
	defer reload.Stop()
	for {
		select {
		case <-ctx.Done():
			return 0
		case <-status.C:
			printMembers(node)
		case <-reload.C:
			if *crlPath != "" {
				publishIfNewer(ctx, node, n, *crlPath)
			}
		}
	}
}

func parseIsolation(s string) (isolate.Isolation, error) {
	switch s {
	case "strict":
		return isolate.StrictIsolation(), nil
	case "ns":
		return isolate.Isolation{Namespaces: true}, nil
	case "process":
		return isolate.Isolation{}, nil
	}
	return isolate.Isolation{}, fmt.Errorf("unknown isolation %q", s)
}

// publishIfNewer spreads the revocation list file when the operator has
// replaced it with a newer one.
func publishIfNewer(ctx context.Context, node *mesh.Node, n *membership.Network, path string) {
	s, err := membership.LoadCRL(path)
	if err != nil {
		return
	}
	c, err := n.VerifyCRL(s)
	if err != nil || c.Seq <= node.CRLSeq() {
		return
	}
	if err := node.PublishCRL(ctx, s); err != nil {
		slog.Warn("publishing revocation list", "err", err)
		return
	}
	fmt.Printf("published revocation list %d\n", c.Seq)
}

func printMembers(node *mesh.Node) {
	ms := node.Members()
	connected := 0
	for _, m := range ms {
		if m.Connected {
			connected++
		}
	}
	fmt.Printf("%s members: %d connected, %d known\n", time.Now().Format(time.TimeOnly), connected, len(ms))
	for _, m := range ms {
		state := "disconnected"
		if m.Connected {
			path := "direct"
			if m.Relayed {
				path = "relayed"
			}
			state = fmt.Sprintf("%s rtt %s via %s", path, m.RTT.Round(100*time.Microsecond), m.Addr)
		}
		fmt.Printf("  %s %-24s %s\n", m.ID, strings.Join(m.Roles, ","), state)
	}
}
