//go:build linux

// Package cli is the network map: `rakkan-node map`. It collects what a node
// can see of its own networks, shows it, and, so that only the holder of the
// network's root key sees the whole network, reports it sealed to that key.
package cli

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/netip"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/collect"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/export"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/rkm"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/share"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/web"
)

const usage = `rakkan-node map — mergeable network maps

usage: rakkan-node map <command> [flags]

observe:
  scan     active scan: ping sweep, mDNS/SSDP queries, TCP probes, SSH host keys;
           wired topology via --snmp (switch FDB, LLDP, ARP) and L3 paths via --trace
  watch    passive observer: listens only, never probes (raw capture if permitted)

inspect:
  show [ID]   summary by network, or everything about one device and why
  locate      set where the observer or a device is (here / set / gps / list)
  class       device types (desktop, laptop, phone, tv, iot, ...): list, or set one by hand
  serve       web viewer: graph + map, filters, time travel (http://127.0.0.1:8742)
  stats       store statistics

the network (only the holder of the root key sees all of it):
  publish  seal this observer's observations to the root key and hand them to the node
  collect  root only: open the sealed reports with the root key into the network store

exchange (all merging is set union: order and repetition do not matter):
  export   write the store as .rkm (or --dot / --json views)
  import   add .rkm files to the store
  merge    merge .rkm files into one, without touching the store
  verify   check signatures and digest of .rkm files (no files: the whole store)
  trust    who you believe: list, set KEY|NAME blocked|unknown|trusted, rm
  key      this observer's key; rotate (routine or after compromise)
  asn      import IP→AS reference data (ip2asn TSV) to label the internet layer
  id       print this observer's public key

global flags (after the command): --db PATH  --key PATH
`

type globals struct{ db, key string }

func (g *globals) register(fs *flag.FlagSet) {
	fs.StringVar(&g.db, "db", store.DefaultPath(), "working store")
	fs.StringVar(&g.key, "key", obs.DefaultKeyPath(), "observer key")
}

// Main runs `rakkan-node map ARGS` and returns the exit code.
func Main(args []string) int {
	if len(args) < 1 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Print(usage)
		return 0
	}
	cmds := map[string]func([]string) error{
		"scan": cmdScan, "watch": cmdWatch, "show": cmdShow, "stats": cmdStats,
		"locate": cmdLocate, "class": cmdClass, "serve": cmdServe, "export": cmdExport, "asn": cmdASN, "trust": cmdTrust, "key": cmdKey, "import": cmdImport, "merge": cmdMerge, "verify": cmdVerify, "id": cmdID, "publish": cmdPublish, "collect": cmdCollect,
	}
	f, ok := cmds[args[0]]
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", args[0], usage)
		return 2
	}
	if err := f(args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "rakkan-node map:", err)
		return 1
	}
	return 0
}

func loadKey(path string) (ed25519.PrivateKey, error) {
	k, created, err := obs.LoadOrCreateKey(path)
	if err == nil && created {
		fmt.Fprintf(os.Stderr, "created observer key %s\n", path)
	}
	return k, err
}

func save(st *store.Store, list []*obs.Observation, out string) error {
	added, err := st.Add(list)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "stored %d observations (%d new)\n", len(list), added)
	if out != "" {
		return writeRKM(out, list)
	}
	return nil
}

func writeRKM(path string, list []*obs.Observation) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := rkm.Write(f, list, "rakkan"); err != nil {
		f.Close()
		return err
	}
	unique := map[[32]byte]bool{}
	for _, o := range list {
		unique[o.ID] = true
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d observations)\n", path, len(unique)) // after dedup, as in the file
	return f.Close()
}

func describeNets(nets []collect.NetCtx) {
	for _, n := range nets {
		gw := "unknown gateway → observer-local net id"
		if n.GWMAC != nil {
			gw = "gw " + n.GW.String() + " " + n.GWMAC.String()
		}
		fmt.Fprintf(os.Stderr, "network %s on %s (%s) %s, %s\n", n.NetID, n.Iface.Name, n.Kind, n.Prefix, gw)
	}
}

func filterIfaces(nets []collect.NetCtx, names string) []collect.NetCtx {
	if names == "" {
		return nets
	}
	want := strings.Split(names, ",")
	return slices.DeleteFunc(nets, func(n collect.NetCtx) bool { return !slices.Contains(want, n.Iface.Name) })
}

func parseAddrs(s string) ([]netip.Addr, error) {
	if s == "" {
		return nil, nil
	}
	var out []netip.Addr
	for _, p := range strings.Split(s, ",") {
		a, err := netip.ParseAddr(strings.TrimSpace(p))
		if err != nil {
			return nil, fmt.Errorf("bad address %q (use an IP)", p)
		}
		out = append(out, a)
	}
	return out, nil
}

func parsePorts(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || n < 1 || n > 65535 {
			return nil, fmt.Errorf("bad port %q", p)
		}
		out = append(out, n)
	}
	return out, nil
}

func cmdScan(args []string) error {
	var g globals
	fs := flag.NewFlagSet("scan", flag.ExitOnError)
	g.register(fs)
	cidr := fs.String("cidr", "", "range to sweep (default: own subnet, or own /24 if larger than --max-hosts)")
	maxHosts := fs.Int("max-hosts", 1024, "largest subnet swept in full")
	ifaces := fs.String("iface", "", "comma-separated interfaces (default: all)")
	ports := fs.String("ports", "", "TCP ports to probe (default: a small built-in set)")
	noProbe := fs.Bool("no-probe", false, "skip TCP probing")
	rate := fs.Duration("rate", 2*time.Millisecond, "delay between ping probes")
	out := fs.String("o", "", "also write this scan to a .rkm file")
	noWifi := fs.Bool("no-wifi", false, "skip the Wi-Fi access point scan")
	noDiscover := fs.Bool("no-discover", false, "skip sweep/mDNS/SSDP/rDNS/TCP; only poll --snmp and --trace targets")
	snmpTargets := fs.String("snmp", "", "comma-separated switches/routers to poll over SNMP (devices you manage)")
	community := fs.String("snmp-community", os.Getenv("RAKKAN_SNMP_COMMUNITY"), "SNMP v2c community (env RAKKAN_SNMP_COMMUNITY)")
	v3user := fs.String("snmp-v3-user", os.Getenv("RAKKAN_SNMP_V3_USER"), "SNMP v3 user (env RAKKAN_SNMP_V3_USER)")
	edgeMax := fs.Int("snmp-edge-max", 4, "switch ports with more learned MACs are treated as uplinks")
	traceTargets := fs.String("trace", "", "comma-separated destinations to traceroute (L3 path)")
	traceHops := fs.Int("trace-hops", 30, "maximum traceroute hops")
	noWAN := fs.Bool("no-wan", false, "do not ask the gateway for its external address")
	stun := fs.String("stun", "", "also ask this STUN server (host:port) for the address seen from the internet")
	fs.Parse(args)

	snmpAddrs, err := parseAddrs(*snmpTargets)
	if err != nil {
		return err
	}
	traceAddrs, err := parseAddrs(*traceTargets)
	if err != nil {
		return err
	}
	snmpAuth := collect.SNMPAuth{Community: *community, V3User: *v3user,
		V3Auth: os.Getenv("RAKKAN_SNMP_V3_AUTH"), V3Priv: os.Getenv("RAKKAN_SNMP_V3_PRIV")}
	if len(snmpAddrs) > 0 && !snmpAuth.Valid() {
		return errors.New("--snmp needs credentials: --snmp-community / RAKKAN_SNMP_COMMUNITY or --snmp-v3-user")
	}

	key, err := loadKey(g.key)
	if err != nil {
		return err
	}
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	s := collect.NewSession(key, os.Stderr)
	nets, err := collect.Discover(s.Pub, true)
	if err != nil {
		return err
	}
	nets = filterIfaces(nets, *ifaces)
	if len(nets) == 0 {
		return errors.New("no usable networks")
	}
	describeNets(nets)

	var prefix netip.Prefix
	if *cidr != "" {
		if prefix, err = netip.ParsePrefix(*cidr); err != nil {
			return err
		}
	}
	portList := collect.DefaultPorts
	if *ports != "" {
		if portList, err = parsePorts(*ports); err != nil {
			return err
		}
	}
	run := func(c collect.Collector) {
		if err := c.Run(ctx, s, nets); err != nil && ctx.Err() == nil {
			s.Log.Printf("%s: %v", c.Name(), err)
		}
	}
	run(collect.Local{})
	if !*noWifi {
		run(collect.WifiScan{Refresh: true, Wait: 8 * time.Second, MaxAge: 2 * time.Minute})
	}
	if !*noDiscover {
		run(collect.Sweep{CIDR: prefix, MaxHosts: *maxHosts, Rate: *rate})
		var wg sync.WaitGroup
		for _, c := range []collect.Collector{collect.MDNSQuery{Wait: 2 * time.Second}, collect.SSDPSearch{Wait: 3 * time.Second}} {
			wg.Add(1)
			go func() { defer wg.Done(); run(c) }()
		}
		wg.Wait()
		targets := func() map[*collect.NetCtx][]netip.Addr { return s.KnownIPs(nets) }
		run(collect.RDNS{Targets: targets})
		if !*noProbe {
			run(collect.Probe{Ports: portList, Targets: targets, Concurrency: 128, Timeout: 800 * time.Millisecond})
		}
	}
	if !*noWAN {
		run(collect.WAN{STUN: *stun, Timeout: 2 * time.Second})
	}
	run(collect.SNMP{Targets: snmpAddrs, Auth: snmpAuth, EdgeMax: *edgeMax, Timeout: 2 * time.Second})
	run(collect.Trace{Targets: traceAddrs, MaxHops: *traceHops, Timeout: time.Second})
	return save(st, s.Flush(), *out)
}

func cmdWatch(args []string) error {
	var g globals
	fs := flag.NewFlagSet("watch", flag.ExitOnError)
	g.register(fs)
	dur := fs.Duration("duration", 0, "stop after this long (default: until Ctrl-C)")
	var o watchOptions
	fs.DurationVar(&o.flush, "flush", 5*time.Minute, "write aggregated observations this often")
	fs.DurationVar(&o.poll, "poll", 30*time.Second, "kernel neighbor table poll interval")
	fs.StringVar(&o.ifaces, "iface", "", "comma-separated interfaces (default: all)")
	fs.StringVar(&o.out, "o", "", "on exit, also write everything observed in this session to a .rkm file")
	fs.Parse(args)

	key, err := loadKey(g.key)
	if err != nil {
		return err
	}
	if st, err := store.Open(g.db); err != nil { // fail early on a bad path
		return err
	} else {
		st.Close()
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *dur > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *dur)
		defer cancel()
	}
	return watch(ctx, g, key, o)
}

type watchOptions struct {
	flush, poll time.Duration
	ifaces, out string
	quiet       bool // no log lines: the daemon runs it for as long as it lives
}

// watch observes passively until ctx ends and stores what it saw every
// flush. It listens and reads the kernel's tables and never sends a probe.
// The store is opened for each flush only, and nothing is kept in memory
// beyond the aggregates of one flush period, unless o.out asks for the
// session's observations.
func watch(ctx context.Context, g globals, key ed25519.PrivateKey, o watchOptions) error {
	logw := io.Writer(os.Stderr)
	if o.quiet {
		logw = io.Discard
	}
	s := collect.NewSession(key, logw)
	var all []*obs.Observation
	doFlush := func() error {
		list := s.Flush()
		if len(list) == 0 {
			return nil
		}
		if o.out != "" {
			all = append(all, list...)
		}
		st, err := store.Open(g.db)
		if err != nil {
			return err
		}
		defer st.Close()
		_, err = st.Add(list)
		s.Log.Printf("flushed %d observations", len(list))
		return err
	}

	var (
		nets       []collect.NetCtx
		listenStop context.CancelFunc = func() {}
		listenWG   sync.WaitGroup
	)
	refresh := func() error {
		fresh, err := collect.Discover(s.Pub, false) // passive: never provoke ARP
		if err != nil {
			return err
		}
		fresh = filterIfaces(fresh, o.ifaces)
		same := len(fresh) == len(nets)
		for i := range fresh {
			if !same || fresh[i].NetID != nets[i].NetID || fresh[i].Iface.Index != nets[i].Iface.Index {
				same = false
			}
		}
		if same {
			return nil
		}
		listenStop()
		listenWG.Wait()
		nets = fresh
		if !o.quiet {
			describeNets(nets)
		}
		collect.Local{}.Run(ctx, s, nets)
		var lctx context.Context
		lctx, listenStop = context.WithCancel(ctx)
		for _, c := range []collect.Collector{collect.MDNSListen{}, collect.SSDPListen{}, collect.Sniff{}} {
			listenWG.Add(1)
			go func() {
				defer listenWG.Done()
				err := c.Run(lctx, s, nets)
				switch {
				case errors.Is(err, collect.ErrNoRaw):
					s.Log.Printf("sniff: %v — ARP/DHCP/LLDP capture off, mDNS/SSDP/neighbor table still on", err)
				case err != nil && lctx.Err() == nil:
					s.Log.Printf("%s: %v", c.Name(), err)
				}
			}()
		}
		return nil
	}
	if err := refresh(); err != nil {
		return err
	}
	if !o.quiet {
		fmt.Fprintln(os.Stderr, "watching passively; Ctrl-C to stop")
	}
	wifiScan := collect.WifiScan{MaxAge: 2 * time.Minute} // passive: read the OS scan cache only
	wifiScan.Run(ctx, s, nets)
	pollT, flushT, netT := time.NewTicker(o.poll), time.NewTicker(o.flush), time.NewTicker(time.Minute)
	defer pollT.Stop()
	defer flushT.Stop()
	defer netT.Stop()
	collect.Neigh{Mode: obs.ModePassive}.Run(ctx, s, nets)
loop:
	for {
		select {
		case <-ctx.Done():
			break loop
		case <-pollT.C:
			collect.Neigh{Mode: obs.ModePassive}.Run(ctx, s, nets)
			wifiScan.Run(ctx, s, nets)
		case <-netT.C:
			if err := refresh(); err != nil {
				s.Log.Printf("refresh networks: %v", err)
			}
		case <-flushT.C:
			if err := doFlush(); err != nil {
				listenStop()
				listenWG.Wait()
				return err
			}
		}
	}
	listenStop()
	listenWG.Wait()
	if err := doFlush(); err != nil {
		return err
	}
	if !o.quiet {
		fmt.Fprintf(os.Stderr, "session: %d observations\n", len(all))
	}
	if o.out != "" {
		return writeRKM(o.out, all)
	}
	return nil
}

func graphFrom(g *globals, files []string) (*resolve.Graph, error) {
	var list []*obs.Observation
	if len(files) > 0 {
		for _, f := range files {
			r, err := readRKM(f)
			if err != nil {
				return nil, err
			}
			list = append(list, r.Observations...)
		}
	} else {
		st, err := store.Open(g.db)
		if err != nil {
			return nil, err
		}
		defer st.Close()
		if list, err = st.All(); err != nil {
			return nil, err
		}
	}
	opts, done := viewerOptions(g)
	defer done()
	return resolve.BuildWith(list, opts), nil
}

// viewerOptions: the local trust book (with this installation's key as
// self) and, if imported, the IP→AS reference table. Call done() after
// building: the AS lookup reads the store.
func viewerOptions(g *globals) (opts resolve.Options, done func()) {
	book := trust.NewBook()
	done = func() {}
	if st, err := store.Open(g.db); err == nil {
		if b, err := st.TrustBook(); err == nil {
			book = b
		}
		opts.ASN = asnFunc(st)
		done = func() { st.Close() }
	}
	if k, _, err := obs.LoadOrCreateKey(g.key); err == nil {
		book.WithSelf(k.Public().(ed25519.PublicKey))
	}
	opts.Trust = book
	return opts, done
}

func asnFunc(st *store.Store) func(netip.Addr) (resolve.AS, bool) {
	look := st.ASNLookup()
	if look == nil {
		return nil
	}
	return func(a netip.Addr) (resolve.AS, bool) {
		as, ok := look(a)
		return resolve.AS{Number: as.Number, Name: as.Name, Country: as.Country}, ok
	}
}

func cmdShow(args []string) error {
	var g globals
	fs := flag.NewFlagSet("show", flag.ExitOnError)
	g.register(fs)
	asJSON := fs.Bool("json", false, "print the resolved graph as JSON")
	from := fs.String("from", "", "comma-separated .rkm files to show instead of the store")
	fs.Parse(args)
	var files []string
	if *from != "" {
		files = strings.Split(*from, ",")
	}
	gr, err := graphFrom(&g, files)
	if err != nil {
		return err
	}
	if id := fs.Arg(0); id != "" {
		d := gr.Device(id)
		if d == nil {
			return fmt.Errorf("no device %q", id)
		}
		if *asJSON {
			return export.JSON(os.Stdout, &resolve.Graph{Devices: []*resolve.Device{d}})
		}
		export.Explain(os.Stdout, gr, d)
		return nil
	}
	if *asJSON {
		return export.JSON(os.Stdout, gr)
	}
	export.Text(os.Stdout, gr)
	return nil
}

func cmdStats(args []string) error {
	var g globals
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	g.register(fs)
	fs.Parse(args)
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	s, err := st.Stats()
	if err != nil {
		return err
	}
	fmt.Printf("store %s\n  observations %d from %d observer(s)\n", g.db, s.Observations, s.Observers)
	for _, m := range []obs.Mode{obs.ModeActive, obs.ModePassive, obs.ModeLocal} {
		fmt.Printf("  %-8s %d\n", m, s.ByMode[m])
	}
	return nil
}

func cmdExport(args []string) error {
	var g globals
	fs := flag.NewFlagSet("export", flag.ExitOnError)
	g.register(fs)
	out := fs.String("o", "", "output .rkm file")
	dot := fs.Bool("dot", false, "print Graphviz DOT of the resolved graph")
	asJSON := fs.Bool("json", false, "print the resolved graph as JSON")
	geojson := fs.Bool("geojson", false, "print located devices as GeoJSON points")
	only := fs.String("only", "all", "whose observations to export: all, trusted, own")
	since := fs.String("since", "", "only observations newer than this (7d, 36h, RFC 3339)")
	location := fs.String("location", "keep", "positions: keep, drop, coarse:1km")
	wanFlag := fs.String("wan", "keep", "external addresses of your networks: keep, drop")
	fs.Parse(args)
	if *dot || *asJSON || *geojson {
		gr, err := graphFrom(&g, nil)
		if err != nil {
			return err
		}
		switch {
		case *dot:
			export.DOT(os.Stdout, gr)
			return nil
		case *geojson:
			return export.GeoJSON(os.Stdout, gr)
		}
		return export.JSON(os.Stdout, gr)
	}
	if *out == "" {
		return errors.New("export: need -o FILE.rkm, --dot, --json or --geojson")
	}
	policy, err := policyFrom(*only, *since, *location, *wanFlag)
	if err != nil {
		return err
	}
	key, err := loadKey(g.key)
	if err != nil {
		return err
	}
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	list, err := st.All()
	if err != nil {
		return err
	}
	book, err := st.TrustBook()
	if err != nil {
		return err
	}
	shared, rep, err := share.Apply(list, policy, key, book)
	if err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, rep)
	return writeRKM(*out, shared)
}

// policyFrom parses the sharing flags common to export and sync.
func policyFrom(only, since, location, wan string) (share.Policy, error) {
	var p share.Policy
	var err error
	switch wan {
	case "", "keep":
	case "drop":
		p.DropWAN = true
	default:
		return p, fmt.Errorf("unknown --wan %q (keep, drop)", wan)
	}
	if p.Who, err = share.ParseWho(only); err != nil {
		return p, err
	}
	if p.Location, err = share.ParseLocation(location); err != nil {
		return p, err
	}
	if since != "" {
		t, err := parseWhen(since)
		if err != nil {
			return p, err
		}
		p.Since = t.UnixMilli()
	}
	return p, nil
}

// parseWhen accepts RFC 3339 or "how long ago": 90m, 36h, 7d.
func parseWhen(s string) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	mult := time.Duration(1)
	if strings.HasSuffix(s, "d") {
		s, mult = strings.TrimSuffix(s, "d")+"h", 24
	}
	d, err := time.ParseDuration(s)
	if err != nil || d < 0 {
		return time.Time{}, fmt.Errorf("bad time %q (RFC 3339 or a duration ago: 90m, 36h, 7d)", s)
	}
	return time.Now().Add(-d * mult), nil
}

const sharingFlagsHelp = `
sharing: --only all|trusted|own   whose observations (blocked keys never leave)
         --since 7d|RFC3339       only recent observations
         --wan keep|drop                  external (public) addresses of your networks
         --location keep|drop|coarse:1km
             signed observations of others cannot be edited: with drop/coarse
             those carrying positions are withheld; your own are re-signed reduced`

func readRKM(path string) (*rkm.Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r, err := rkm.Read(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(r.Rejected) > 0 {
		fmt.Fprintf(os.Stderr, "%s: %d invalid record(s) dropped (first: %v)\n", path, len(r.Rejected), r.Rejected[0])
	}
	return r, nil
}

func cmdImport(args []string) error {
	var g globals
	fs := flag.NewFlagSet("import", flag.ExitOnError)
	g.register(fs)
	fs.Parse(args)
	if fs.NArg() == 0 {
		return errors.New("import: no files")
	}
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	for _, f := range fs.Args() {
		r, err := readRKM(f)
		if err != nil {
			return err
		}
		added, err := st.Add(r.Observations)
		if err != nil {
			return err
		}
		fmt.Printf("%s: %d observations, %d new\n", f, len(r.Observations), added)
	}
	return nil
}

func cmdMerge(args []string) error {
	fs := flag.NewFlagSet("merge", flag.ExitOnError)
	out := fs.String("o", "", "output .rkm file")
	fs.Parse(args)
	if *out == "" || fs.NArg() == 0 {
		return errors.New("usage: rakkan-node map merge -o OUT.rkm A.rkm B.rkm ...")
	}
	var all []*obs.Observation
	for _, f := range fs.Args() {
		r, err := readRKM(f)
		if err != nil {
			return err
		}
		all = append(all, r.Observations...)
	}
	return writeRKM(*out, all)
}

func cmdVerify(args []string) error {
	if len(args) == 0 || args[0][0] == '-' { // no files: the local store
		var g globals
		fs := flag.NewFlagSet("verify", flag.ExitOnError)
		g.register(fs)
		fs.Parse(args)
		st, err := store.Open(g.db)
		if err != nil {
			return err
		}
		defer st.Close()
		ok, nbad, err := st.Verify()
		if err != nil {
			return err
		}
		fmt.Printf("store %s: %d valid, %d invalid signature(s) or encoding(s)\n", g.db, ok, nbad)
		if nbad > 0 {
			return errors.New("the store was modified outside rakkan")
		}
		return nil
	}
	bad := false
	for _, f := range args {
		r, err := readRKM(f)
		if err != nil {
			return err
		}
		ids := make([][32]byte, len(r.Observations))
		observers := map[string]bool{}
		modes := map[obs.Mode]int{}
		for i, o := range r.Observations {
			ids[i] = o.ID
			observers[hex.EncodeToString(o.Observer)[:12]] = true
			modes[o.Mode]++
		}
		d := rkm.SetDigest(ids)
		status := "OK"
		if !r.DigestOK {
			status, bad = "DIGEST MISMATCH", true
		}
		fmt.Printf("%s: %s  %d obs (active %d, passive %d, local %d) from %d observer(s), created %s\n  set digest %x\n",
			f, status, len(r.Observations), modes[obs.ModeActive], modes[obs.ModePassive], modes[obs.ModeLocal],
			len(observers), r.Created.Format(time.DateTime), d[:16])
	}
	if bad {
		return errors.New("verification failed")
	}
	return nil
}

func cmdID(args []string) error {
	var g globals
	fs := flag.NewFlagSet("id", flag.ExitOnError)
	g.register(fs)
	fs.Parse(args)
	k, err := loadKey(g.key)
	if err != nil {
		return err
	}
	fmt.Println(obs.RKMID(k.Public().(ed25519.PublicKey)))
	return nil
}

// cmdLocate records and lists where things are. Positions are claims like
// any other: signed, mergeable, and judged against other sources by the
// resolver (spec §9).
func cmdLocate(args []string) error {
	const help = `usage: rakkan-node map locate <command>

  here --lat N --lon N [--radius M] [--logical TEXT] [--region TEXT]
        where this observer is now
  set DEVICE --lat N --lon N [--radius M] [--logical TEXT] [--region TEXT]
        where a device is; DEVICE is an id from "rakkan-node map show" or an identifier (mac:..., bssid:...)
  gps [--gpsd HOST:PORT]
        take one fix from a running gpsd as the observer's position
  list  located devices

Without --radius, coordinates are taken as accurate to 25 m. Positions are
sensitive: they travel with the map when you export it.
`
	if len(args) == 0 {
		fmt.Print(help)
		return nil
	}
	sub, rest := args[0], args[1:]
	var g globals
	fs := flag.NewFlagSet("locate "+sub, flag.ExitOnError)
	g.register(fs)
	lat := fs.Float64("lat", math.NaN(), "latitude, degrees")
	lon := fs.Float64("lon", math.NaN(), "longitude, degrees")
	radius := fs.Float64("radius", 25, "accuracy: radius of the error circle, metres")
	logical := fs.String("logical", "", "place without coordinates, e.g. \"HQ/2/R201\"")
	region := fs.String("region", "", "country or city")
	gpsdAddr := fs.String("gpsd", "127.0.0.1:2947", "gpsd address")
	target := ""
	if sub == "set" {
		if len(rest) == 0 || strings.HasPrefix(rest[0], "-") {
			return errors.New("locate set: need a device first: rakkan-node map locate set DEVICE --lat ... --lon ...")
		}
		target, rest = rest[0], rest[1:]
	}
	fs.Parse(rest)

	switch sub {
	case "list":
		gr, err := graphFrom(&g, nil)
		if err != nil {
			return err
		}
		n := 0
		for _, d := range gr.Devices {
			if d.Loc.Level == resolve.LevelUnknown && d.Loc.Logical == "" {
				continue
			}
			n++
			fmt.Printf("%s  %-28s %s\n", d.ID[:7], d.Label, locLine(d.Loc))
		}
		fmt.Printf("%d of %d devices have a known place\n", n, len(gr.Devices))
		return nil
	case "here", "set", "gps":
	default:
		fmt.Print(help)
		return fmt.Errorf("unknown locate command %q", sub)
	}

	key, err := loadKey(g.key)
	if err != nil {
		return err
	}
	s := collect.NewSession(key, os.Stderr)
	hasCoords := !math.IsNaN(*lat) || !math.IsNaN(*lon)
	if hasCoords && (math.IsNaN(*lat) || math.IsNaN(*lon)) {
		return errors.New("give both --lat and --lon")
	}
	place := collect.Place{Logical: *logical, Region: *region}
	if hasCoords {
		place.Lat, place.Lon, place.RadiusM = *lat, *lon, *radius
	}

	switch sub {
	case "gps":
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		if err := (collect.GPSD{Addr: *gpsdAddr, Timeout: 15 * time.Second}).Run(ctx, s, nil); err != nil {
			return err
		}
	case "here":
		if err := collect.EmitPlace(s, []string{obs.RKMID(s.Pub)}, place); err != nil {
			return err
		}
	case "set":
		subject, err := locateSubject(&g, target)
		if err != nil {
			return err
		}
		if err := collect.EmitPlace(s, subject, place); err != nil {
			return err
		}
	}
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	return save(st, s.Flush(), "")
}

func cmdClass(args []string) error {
	const help = `usage: rakkan-node map class [DEVICE CLASS] [flags]

Without arguments, lists devices by inferred type and why. With DEVICE and
CLASS, records the type by hand; it outweighs inferred evidence.

  DEVICE  an id from "rakkan-node map show" or an identifier (mac:..., ssh:...)
  CLASS   one of: %s
`
	var g globals
	fs := flag.NewFlagSet("class", flag.ExitOnError)
	g.register(fs)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, help, strings.Join(resolve.Classes, ", ")); fs.PrintDefaults() }
	pos := parseMixed(fs, args)
	switch len(pos) {
	case 0:
		gr, err := graphFrom(&g, nil)
		if err != nil {
			return err
		}
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		counts := map[string]int{}
		for _, d := range gr.Devices {
			kind, conf, why := "unknown", "", ""
			if c := d.Class; c != nil {
				kind, conf, why = c.Kind, fmt.Sprintf("%.0f%%", c.Conf*100), strings.Join(c.Reasons, "; ")
				if c.Alt != "" {
					why += fmt.Sprintf("  [or %s %.0f%%]", c.Alt, c.AltConf*100)
				}
			}
			counts[kind]++
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", d.ID[:7], truncate(d.Label, 28), kind, conf, why)
		}
		tw.Flush()
		var parts []string
		for _, k := range append(slices.Clone(resolve.Classes), "unknown") {
			if counts[k] > 0 {
				parts = append(parts, fmt.Sprintf("%s %d", k, counts[k]))
			}
		}
		fmt.Printf("\n%d devices: %s\n", len(gr.Devices), strings.Join(parts, ", "))
		return nil
	case 2:
	default:
		fs.Usage()
		return errors.New("class: give both DEVICE and CLASS, or nothing to list")
	}
	kind := strings.ToLower(pos[1])
	if !slices.Contains(resolve.Classes, kind) {
		return fmt.Errorf("unknown class %q; one of: %s", kind, strings.Join(resolve.Classes, ", "))
	}
	subject, err := locateSubject(&g, pos[0])
	if err != nil {
		return err
	}
	key, err := loadKey(g.key)
	if err != nil {
		return err
	}
	s := collect.NewSession(key, os.Stderr)
	s.Emit(obs.Body{Mode: obs.ModeLocal, Method: "manual", Subject: subject, Claims: []obs.Claim{{K: "class", V: kind}}})
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	return save(st, s.Flush(), "")
}

// locateSubject turns a user-given target into identifiers to attach the
// position to: a device id prefix is expanded to its strongest identifiers,
// anything with a "kind:" prefix is used as is.
func locateSubject(g *globals, target string) ([]string, error) {
	if strings.Contains(target, ":") {
		return []string{target}, nil
	}
	gr, err := graphFrom(g, nil)
	if err != nil {
		return nil, err
	}
	d := gr.Device(target)
	if d == nil {
		return nil, fmt.Errorf("no device %q (see rakkan-node map show)", target)
	}
	var ids []string
	for _, id := range d.Idents {
		if obs.StrengthOf(id) > obs.Weak && len(ids) < 3 {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 && len(d.Idents) > 0 {
		ids = d.Idents[:1] // an address-only device: the address is all there is
	}
	fmt.Fprintf(os.Stderr, "device %s %q → %s\n", d.ID[:7], d.Label, strings.Join(ids, ", "))
	return ids, nil
}

func locLine(l *resolve.Location) string {
	var parts []string
	if l.Point != nil {
		s := fmt.Sprintf("%s %.5f,%.5f ±%.0fm %.0f%%", l.Level, l.Point.Lat, l.Point.Lon, l.RadiusM, l.Conf*100)
		if l.Derived {
			s += " (carried over links)"
		}
		if l.Conflict {
			s += " (sources disagree)"
		}
		parts = append(parts, s)
	}
	if l.Region != "" {
		parts = append(parts, l.Region)
	}
	if l.Logical != "" {
		parts = append(parts, "“"+l.Logical+"”")
	}
	return strings.Join(parts, "  ")
}

const osmTiles = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

type storeSource struct{ st *store.Store }

func (s storeSource) Version() (string, error) {
	st, err := s.st.Stats()
	return strconv.Itoa(st.Observations), err
}
func (s storeSource) All() ([]*obs.Observation, error) { return s.st.All() }

type fileSource struct{ list []*obs.Observation }

func (f fileSource) Version() (string, error)         { return "files", nil }
func (f fileSource) All() ([]*obs.Observation, error) { return f.list, nil }

func cmdServe(args []string) error {
	var g globals
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	g.register(fs)
	addr := fs.String("addr", "127.0.0.1:8742", "listen address (keep it on loopback: the map is sensitive)")
	from := fs.String("from", "", "comma-separated .rkm files to view instead of the store")
	tiles := fs.String("tiles", osmTiles, `basemap tile URL template, or "off" (tiles reveal the viewed area to the tile server)`)
	allow := fs.String("allow-host", "", "extra Host names to accept when serving beyond this machine")
	fs.Parse(args)

	var src web.Source
	if *from != "" {
		var list []*obs.Observation
		for _, f := range strings.Split(*from, ",") {
			r, err := readRKM(f)
			if err != nil {
				return err
			}
			list = append(list, r.Observations...)
		}
		src = fileSource{list}
	} else {
		st, err := store.Open(g.db)
		if err != nil {
			return err
		}
		defer st.Close()
		src = storeSource{st}
	}
	if *tiles == "off" {
		*tiles = ""
	}
	viewStore, err := store.Open(g.db) // trust book and AS table, re-read per build
	if err != nil {
		return err
	}
	defer viewStore.Close()
	own, err := loadKey(g.key)
	if err != nil {
		return err
	}
	srv := &web.Server{Src: src, Tiles: *tiles, Options: func() (resolve.Options, string) {
		book, err := viewStore.TrustBook()
		if err != nil {
			book = trust.NewBook()
		}
		book.WithSelf(own.Public().(ed25519.PublicKey))
		o := resolve.Options{Trust: book, ASN: asnFunc(viewStore)}
		v := ""
		for _, e := range o.Trust.Entries() {
			v += e.Key + e.Level.String()
		}
		if o.ASN != nil {
			v += "+asn"
		}
		return o, fmt.Sprintf("%x", sha256.Sum256([]byte(v)))
	}}
	if *allow != "" {
		srv.AllowedHosts = strings.Split(*allow, ",")
	}
	host, _, err := net.SplitHostPort(*addr)
	if err != nil {
		return err
	}
	if ip := net.ParseIP(host); host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		fmt.Fprintf(os.Stderr, "warning: serving on %s makes the map (devices, addresses, positions) visible to the network\n", *addr)
		if *allow == "" {
			fmt.Fprintln(os.Stderr, "         requests will be refused unless their Host is listed in --allow-host")
		}
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "rakkan viewer on http://%s/  (Ctrl-C to stop)\n", ln.Addr())
	if *tiles != "" {
		fmt.Fprintln(os.Stderr, "basemap tiles from", *tiles, "(use --tiles off to keep everything local)")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	hs := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() { <-ctx.Done(); hs.Close() }()
	if err := hs.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func cmdASN(args []string) error {
	const help = `usage: rakkan-node map asn import FILE[.gz]

FILE is an IP-to-AS table in the ip2asn TSV format
(range_start, range_end, AS_number, country_code, AS_description), e.g.
ip2asn-combined.tsv.gz from https://iptoasn.com (public domain). It is
reference data: kept locally, used to label public addresses with their
autonomous system, never put into maps. Importing replaces the previous table.
`
	var g globals
	fs := flag.NewFlagSet("asn", flag.ExitOnError)
	g.register(fs)
	var pos []string
	for {
		fs.Parse(args)
		if fs.NArg() == 0 {
			break
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
	if len(pos) != 2 || pos[0] != "import" {
		fmt.Print(help)
		return errors.New("usage: rakkan-node map asn import FILE")
	}
	f, err := os.Open(pos[1])
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	n, err := st.ImportASN(f)
	if err != nil {
		return err
	}
	fmt.Printf("imported %d routed ranges\n", n)
	return nil
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n-1]) + "…"
	}
	return s
}
