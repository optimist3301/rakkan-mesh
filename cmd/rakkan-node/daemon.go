package main

import (
	"context"
	"runtime"

	"errors"
	"flag"
	"fmt"
	"github.com/optimist3301/rakkan-mesh/internal/api"
	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/exec"
	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/llm"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	"github.com/optimist3301/rakkan-mesh/internal/trust"
	"io/fs"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/web"
	"github.com/optimist3301/rakkan-mesh/internal/ygg"
)

func daemonCmd(args []string) int {
	flags := flag.NewFlagSet("daemon", flag.ExitOnError)
	configPath := flags.String("config", "", "JSON config with the node's settings, private key and network certificate; command-line flags override it")
	network := flags.String("network", "", "network ID (required)")
	keyPath := flags.String("key", "node.key", "node private key")
	certPath := flags.String("cert", "node.cert", "node certificate; without one the node joins an open network as a guest")
	proofPath := flags.String("proof", "node.proof", "a guest's proof of work, solved and saved when missing")
	policyPath := flags.String("policy", "", "network policy from 'network open'; reloaded and published to the network when it changes")
	trustPath := flags.String("trust", "", "ledger of peers' credit and reputation (default <data>/trust.json)")
	freeCredit := flags.Float64("free-credit", 50, "work a guest may ask of this node beyond what it did for it, in billions of gas")
	crlPath := flags.String("crl", "", "revocation list; reloaded and published to the network when it changes")
	listen := flags.String("listen", "", "comma-separated listen multiaddrs (default: QUIC and TCP on all interfaces); none: no IP address, only Yggdrasil")
	bootstrap := flags.String("bootstrap", "", "comma-separated multiaddrs of known members, ending in /p2p/<peer ID>")
	peerCache := flags.String("peers", "peers.json", "file remembering members' addresses across restarts; empty disables")
	mdnsOn := flags.Bool("mdns", true, "discover members on the local network")
	mainline := flags.Bool("mainline", true, "find members through the BitTorrent Mainline DHT, and keep the network's records there (network announce)")
	yggOn := flags.Bool("yggdrasil", true, "reach members behind NAT through an embedded Yggdrasil node (no root, no TUN device)")
	yggPeers := flags.String("ygg-peers", "", `comma-separated Yggdrasil peer URIs (tls://host:port, ...); default: a few public ones; "none" for no peers`)
	yggListen := flags.String("ygg-listen", "", "comma-separated URIs to accept Yggdrasil peerings on, e.g. tls://0.0.0.0:7743")
	nat := flags.Bool("nat", true, "UPnP port mapping, hole punching and AutoNAT")
	relay := flags.Bool("relay", false, "relay traffic for members; enable on publicly reachable nodes")
	reach := flags.String("reachability", "auto", "auto (from interface addresses), public, or private (behind NAT)")
	verbose := flags.Bool("v", false, "debug logging")
	workers := flags.Int("workers", runtime.GOMAXPROCS(0), "tasks this node runs for the network at once; 0 only submits tasks")
	isolation := flags.String("isolation", "strict", "worker isolation: strict, ns or process")
	cgroup := flags.String("cgroup", "", `delegated cgroup v2 directory for per-worker limits, or "self"`)
	webAddr := flags.String("web", "127.0.0.1:7780", "address of the web page: members, models and a chat; empty disables")
	webToken := flags.String("web-token", os.Getenv("RAKKAN_WEB_TOKEN"), "token the web page asks for; required when -web is not a loopback address (default $RAKKAN_WEB_TOKEN)")
	apiPath := flags.String("api", "rakkan.sock", "unix socket for the local API used by SDKs; empty disables")
	dataDir := flags.String("data", "data", "directory for stored blobs: large task inputs and outputs")
	storeGB := flags.Float64("store-gb", 10, "disk budget for stored blobs, in GiB")
	maxInputMB := flags.Int("max-input-mb", 16, "largest task input this node accepts, in MiB")
	maxOutputMB := flags.Int("max-output-mb", 16, "largest task output this node accepts, in MiB")
	serveLLM := flags.Bool("llm", false, "serve language model pipeline stages for the network")
	llmMemGB := flags.Float64("llm-mem-gb", 4, "memory for model weights and caches served to the network, in GiB; with a GPU, for caches only")
	gpuFlag := flags.String("gpu", "auto", `GPU for language models: "auto" (the best exact one), "off", a device index, or "list" to show them`)
	gpuMemGB := flags.Float64("gpu-mem-gb", 0, "GPU memory for model weights, in GiB (default: 85% of the device's)")
	hubURL := flags.String("hub", "https://huggingface.co", `Hugging Face Hub for models imported from it ("hub" format); empty: only from members (HF_TOKEN authorizes gated repositories)`)
	statusEvery := flags.Duration("status", 30*time.Second, "how often to print the member table")
	flags.Parse(args)

	var cf *nodeConfig
	if *configPath != "" {
		var err error
		if cf, err = loadConfig(*configPath); err != nil {
			return fail(err)
		}
		if err := cf.apply(flags); err != nil {
			return fail(err)
		}
		// Explicit -key and -cert files win over the embedded ones.
		flags.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "key":
				cf.key = nil
			case "cert":
				cf.cert = nil
			}
		})
	}

	if *gpuFlag == "list" {
		return listGPUs()
	}
	n, err := membership.ParseNetwork(*network)
	if err != nil {
		return fail(fmt.Errorf("-network: %w", err))
	}
	var key crypto.PrivKey
	var cert *membership.SignedCert
	if cf != nil {
		key, cert = cf.key, cf.cert
	}
	if key == nil {
		if key, err = membership.LoadKey(*keyPath); err != nil {
			return fail(err)
		}
	}
	// A config with its own key and no certificate describes a guest.
	if cert == nil && (cf == nil || cf.key == nil) {
		cert, err = membership.LoadCert(*certPath)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return fail(err)
		}
	}
	reachability, err := mesh.ParseReachability(*reach)
	if err != nil {
		return fail(err)
	}
	cfg := mesh.Config{Key: key, Network: n, Cert: cert, MDNS: *mdnsOn, Mainline: *mainline, NAT: *nat, RelayService: *relay, Reachability: reachability, PeerCache: *peerCache}
	if *policyPath != "" {
		if cfg.Policy, err = membership.LoadPolicy(*policyPath); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return fail(err)
		}
	}
	if *crlPath != "" {
		// A missing file is fine: it is picked up once the operator creates it.
		if cfg.CRL, err = membership.LoadCRL(*crlPath); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return fail(err)
		}
	}
	switch *listen {
	case "":
	case "none":
		cfg.ListenAddrs = []string{}
	default:
		cfg.ListenAddrs = strings.Split(*listen, ",")
	}
	var yggCfg *ygg.Config
	if *yggOn {
		cfg.Yggdrasil = true
		switch *yggPeers {
		case "":
		case "none":
			cfg.YggdrasilPeers = []string{}
		default:
			cfg.YggdrasilPeers = strings.Split(*yggPeers, ",")
		}
		if *yggListen != "" {
			cfg.YggdrasilListen = strings.Split(*yggListen, ",")
		}
		yggCfg = &ygg.Config{Peers: cfg.YggdrasilPeers}
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
	if cert == nil {
		id, err := peer.IDFromPrivateKey(key)
		if err != nil {
			return fail(err)
		}
		pr, policy, err := guestProof(context.Background(), n, id, cfg.Policy, policySource{bootstrap: cfg.Bootstrap, mainline: *mainline, ygg: yggCfg}, *proofPath)
		if err != nil {
			return fail(err)
		}
		cfg.Proof, cfg.Policy = &pr, policy
		cfg.OnProof = func(pr membership.Proof) { membership.SaveProof(*proofPath, pr) }
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

	store, err := blob.OpenStore(*dataDir, int64(*storeGB*(1<<30)))
	if err != nil {
		return fail(fmt.Errorf("blob store: %w", err))
	}
	blobs := blob.NewService(node, store, cfg.Logger)
	if *trustPath == "" {
		*trustPath = filepath.Join(*dataDir, "trust.json")
	}
	tr, err := trust.New(trust.Config{Node: node, Path: *trustPath, FreeCredit: int64(*freeCredit * 1e9), Logger: cfg.Logger})
	if err != nil {
		return fail(fmt.Errorf("trust: %w", err))
	}
	defer tr.Close()

	ceiling := sandbox.DefaultLimits()
	ceiling.MaxInputBytes = *maxInputMB << 20
	ceiling.MaxOutputBytes = *maxOutputMB << 20
	svcCfg := exec.Config{Node: node, Blobs: blobs, Ceiling: ceiling, Trust: tr, Logger: cfg.Logger}
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
	dev, err := openGPU(*gpuFlag)
	if err != nil {
		return fail(err)
	}
	if dev != nil {
		defer dev.Close()
	}
	if *serveLLM && os.Getenv("GOMEMLIMIT") == "" {
		// Model layers are large (DeepSeek-V4.1's are 7 GiB a blob), and by
		// default Go lets the heap grow to twice what is live before it
		// collects: one layer let go and the next loaded could hold both.
		// The budget plus room for the rest bounds it.
		debug.SetMemoryLimit(int64(*llmMemGB*(1<<30)) + 2<<30)
	}
	var hub *llm.Hub
	if *hubURL != "" {
		hub = &llm.Hub{Endpoint: *hubURL, Token: os.Getenv("HF_TOKEN")}
	}
	lm, err := llm.NewService(llm.ServiceConfig{Node: node, Blobs: blobs, Serve: *serveLLM,
		MemoryBytes: int64(*llmMemGB * (1 << 30)), Device: dev, DeviceBytes: int64(*gpuMemGB * (1 << 30)), Trust: tr, Logger: cfg.Logger,
		Hub: hub, DataDir: *dataDir})
	if err != nil {
		return fail(err)
	}
	defer lm.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *apiPath != "" {
		l, err := api.Listen(*apiPath)
		if err != nil {
			return fail(fmt.Errorf("api: %w", err))
		}
		defer os.Remove(*apiPath)
		srv := &api.Server{Service: svc, Blobs: store, LLM: lm, Node: node, Trust: tr, Network: n.ID, Version: version}
		go func() {
			if err := srv.Serve(ctx, l); err != nil {
				slog.Error("api", "err", err)
			}
		}()
		fmt.Printf("api on %s\n", *apiPath)
	}
	if *webAddr != "" {
		if err := startWeb(ctx, *webAddr, *webToken, &api.Server{Service: svc, Blobs: store, LLM: lm, Node: node, Trust: tr, Network: n.ID, Version: version}); err != nil {
			return fail(err)
		}
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
			printMembers(node, tr)
		case <-reload.C:
			if *crlPath != "" {
				publishIfNewer(ctx, node, n, *crlPath)
			}
			if *policyPath != "" {
				publishPolicyIfNewer(ctx, node, n, *policyPath)
			}
		}
	}
}

// listGPUs prints the Vulkan devices.
func listGPUs() int {
	devs, err := gpu.Devices()
	if err != nil {
		return fail(err)
	}
	if len(devs) == 0 {
		fmt.Println("no Vulkan devices")
	}
	best, _ := gpu.Best(devs)
	for _, d := range devs {
		mark := " "
		if d.Name == best.Name && d.Index == best.Index {
			mark = "*"
		}
		exact := "exact"
		if !d.Exact {
			exact = "not exact: results may differ from CPU nodes'"
		}
		fmt.Printf("%s %d  %s (%s, Vulkan %s, %.1f GiB, %s)\n", mark, d.Index, d.Name, d.Type, d.API, float64(d.Memory)/(1<<30), exact)
	}
	fmt.Println("* the device -gpu auto picks")
	return 0
}

// openGPU opens the GPU the -gpu flag names, or none.
func openGPU(which string) (*gpu.Device, error) {
	if which == "off" {
		return nil, nil
	}
	devs, err := gpu.Devices()
	if err != nil {
		if which == "auto" {
			slog.Info("no GPU: language models run on the CPU", "reason", err)
			return nil, nil
		}
		return nil, err
	}
	var info gpu.Info
	if which == "auto" {
		var ok bool
		if info, ok = gpu.Best(devs); !ok {
			slog.Info("no suitable GPU: language models run on the CPU")
			return nil, nil
		}
	} else {
		i, err := strconv.Atoi(which)
		if err != nil || i < 0 || i >= len(devs) {
			return nil, fmt.Errorf("-gpu %q: want auto, off, list or a device index", which)
		}
		info = devs[i]
		if !info.Exact {
			slog.Warn("this GPU does not guarantee exact arithmetic: rarely, its results may differ from other nodes' and fail checks", "device", info.Name)
		}
	}
	d, err := gpu.Open(info)
	if err != nil {
		return nil, fmt.Errorf("GPU %s: %w", info.Name, err)
	}
	fmt.Printf("GPU: %s (%s, %.1f GiB)\n", info.Name, info.Type, float64(info.Memory)/(1<<30))
	return d, nil
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

// publishPolicyIfNewer spreads the policy file when the operator has
// replaced it with a newer one.
func publishPolicyIfNewer(ctx context.Context, node *mesh.Node, n *membership.Network, path string) {
	s, err := membership.LoadPolicy(path)
	if err != nil {
		return
	}
	p, err := n.VerifyPolicy(s)
	if cur := node.Policy(); err != nil || cur.Network != "" && p.Seq <= cur.Seq {
		return
	}
	if err := node.PublishPolicy(ctx, s); err != nil {
		slog.Warn("publishing policy", "err", err)
		return
	}
	fmt.Printf("published policy %d (open: %v)\n", p.Seq, p.Open)
}

// startWeb serves the web page. A busy port (another node on the machine)
// only costs the page; an address beyond loopback without a token is an
// error.
func startWeb(ctx context.Context, addr, token string, srv *api.Server) error {
	if !web.IsLoopback(addr) && token == "" {
		return fmt.Errorf("-web %s is reachable from other machines: set -web-token (or RAKKAN_WEB_TOKEN), or use 127.0.0.1", addr)
	}
	l, err := net.Listen("tcp", addr)
	if err != nil {
		slog.Warn("web page disabled", "addr", addr, "err", err)
		return nil
	}
	go func() {
		if err := web.Serve(ctx, l, web.Handler(srv.Handler(), token)); err != nil {
			slog.Error("web", "err", err)
		}
	}()
	host := l.Addr().String()
	if web.IsLoopback(addr) {
		host = strings.Replace(host, "127.0.0.1", "localhost", 1)
	}
	fmt.Printf("web page on http://%s\n", host)
	return nil
}

func printMembers(node *mesh.Node, tr *trust.Service) {
	ms := node.Members()
	connected := 0
	for _, m := range ms {
		if m.Connected {
			connected++
		}
	}
	yggInfo := ""
	if node.YggdrasilAddr() != nil {
		yggInfo = fmt.Sprintf("; yggdrasil: %d peerings up", node.YggdrasilPeers())
	}
	fmt.Printf("%s members: %d connected, %d known%s\n", time.Now().Format(time.TimeOnly), connected, len(ms), yggInfo)
	for _, m := range ms {
		state := "disconnected"
		if m.Connected {
			path := "direct"
			if m.Relayed {
				path = "relayed"
			}
			state = fmt.Sprintf("%s rtt %s via %s", path, m.RTT.Round(100*time.Microsecond), m.Addr)
		}
		if m.Guest {
			state += fmt.Sprintf(", score %.2f", tr.Score(m.ID))
		}
		fmt.Printf("  %s %-24s %s\n", m.ID, strings.Join(m.Roles, ","), state)
	}
}
