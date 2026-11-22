//go:build linux

package cli

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/collector"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/rkm"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/share"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
	"github.com/optimist3301/rakkan-mesh/internal/nodesock"
)

// The map of the whole network is a secret of its root. A member's own maps
// stay in its own store; `publish` hands the observations of this observer to
// the local rakkan-node, which seals them to the network's root key before
// they leave the machine. Members that relay them see only sealed bytes.
// `collect`, run by the holder of the root key, takes the sealed reports from
// any node, opens them and merges them into a store of its own, which every
// other command (show, serve, export) then reads as one map.

// nodeFlags says which node's API to talk to.
type nodeFlags struct{ api, config string }

func (n *nodeFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&n.api, "api", "", "the node's API socket (default: where the installer puts it)")
	fs.StringVar(&n.config, "config", "", "the node's JSON config, to take the socket from its \"api\" field")
}

func (n *nodeFlags) client() (*http.Client, error) {
	sock, err := nodesock.Find(n.api, n.config)
	if err != nil {
		return nil, err
	}
	return nodesock.Client(sock), nil
}

// loop runs f once, or every d until interrupted.
func loop(every time.Duration, f func() error) error {
	if every <= 0 {
		return f()
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	for {
		if err := f(); err != nil {
			fmt.Fprintln(os.Stderr, "rakkan-node map:", err)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(every):
		}
	}
}

// ---------- publish ----------

// apiSender hands reports to a running node over its local API.
func apiSender(client *http.Client) func([]byte) error {
	return func(f []byte) error {
		res, err := client.Post("http://rakkan/v1/reports", "application/octet-stream", bytes.NewReader(f))
		if err != nil {
			return fmt.Errorf("the node: %w", err)
		}
		body, _ := io.ReadAll(io.LimitReader(res.Body, 1<<16))
		res.Body.Close()
		if res.StatusCode != http.StatusOK {
			return fmt.Errorf("the node refused the report (%d): %s", res.StatusCode, bytes.TrimSpace(body))
		}
		return nil
	}
}

// ReporterOptions configure a Reporter.
type ReporterOptions struct {
	DB, Key string        // the observer's store and key (default: the usual ones)
	Every   time.Duration // how often; each round is jittered by up to a tenth
	Scan    bool          // also scan the node's own networks before each report: active, sends probes (see `scan`)
	// Passive is the default and always on: the Reporter listens (mDNS,
	// SSDP, raw capture if permitted) and reads the kernel's neighbour table
	// and Wi-Fi scan cache, as `watch` does, and sends nothing.
	// Keep is how long observations stay in the store (0: DefaultKeep). The
	// store would otherwise grow with every round, for as long as the node runs.
	Keep time.Duration
	// Location and WAN are the --location and --wan of publish.
	Location, WAN string
	// Send hands a report to the network, normally Node.PublishReport.
	Send func([]byte) error
}

// DefaultKeep is how long a node keeps what it observed.
const DefaultKeep = 7 * 24 * time.Hour

// Reporter is the node's own periodic scanner and publisher: the daemon runs
// it inside itself, so it needs no API socket.
type Reporter struct {
	g      globals
	key    ed25519.PrivateKey
	policy share.Policy
	opts   ReporterOptions
}

// NewReporter prepares a Reporter.
func NewReporter(o ReporterOptions) (*Reporter, error) {
	g := globals{db: o.DB, key: o.Key}
	if g.db == "" {
		g.db = store.DefaultPath()
	}
	if g.key == "" {
		g.key = obs.DefaultKeyPath()
	}
	if o.Location == "" {
		o.Location = "keep"
	}
	if o.WAN == "" {
		o.WAN = "keep"
	}
	if o.Keep <= 0 {
		o.Keep = DefaultKeep
	}
	policy, err := policyFrom("own", "", o.Location, o.WAN)
	if err != nil {
		return nil, err
	}
	key, err := loadKey(g.key)
	if err != nil {
		return nil, err
	}
	return &Reporter{g: g, key: key, policy: policy, opts: o}, nil
}

// Run scans, reports and prunes every Every until ctx ends. Between rounds
// nothing is held: the store is opened for a round and closed, and the memory
// a scan used is handed back to the system.
func (r *Reporter) Run(ctx context.Context) {
	go func() {
		// A failed watch (no permission to list interfaces, say) ends
		// the passive part only; reports still go out.
		if err := watch(ctx, r.g, r.key, watchOptions{flush: time.Minute, poll: 30 * time.Second, quiet: true}); err != nil && ctx.Err() == nil {
			fmt.Fprintln(os.Stderr, "rakkan-node map: passive watch:", err)
		}
	}()
	for {
		r.round()
		debug.FreeOSMemory()
		d := r.opts.Every
		d += time.Duration((rand.Float64()*0.2 - 0.1) * float64(d))
		select {
		case <-ctx.Done():
			return
		case <-time.After(d):
		}
	}
}

func (r *Reporter) round() {
	if r.opts.Scan {
		if err := cmdScan([]string{"--db", r.g.db, "--key", r.g.key}); err != nil {
			fmt.Fprintln(os.Stderr, "rakkan-node map: scan:", err)
		}
	}
	if err := publishOnce(r.g, r.opts.Send, r.key, r.policy, false); err != nil {
		fmt.Fprintln(os.Stderr, "rakkan-node map:", err)
	}
	if err := pruneStore(r.g, r.opts.Keep); err != nil {
		fmt.Fprintln(os.Stderr, "rakkan-node map: prune:", err)
	}
}

// pruneStore drops what is older than keep, but not own observations that
// have not been reported yet.
func pruneStore(g globals, keep time.Duration) error {
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	key, _, err := obs.LoadOrCreateKey(g.key)
	if err != nil {
		return err
	}
	own, err := ownObservers(st, key)
	if err != nil {
		return err
	}
	n, err := st.Prune(time.Now().Add(-keep).UnixMilli(), own)
	if n > 0 {
		fmt.Fprintf(os.Stderr, "dropped %d observations older than %s\n", n, keep)
	}
	return err
}

// ownObservers are this installation's keys: the current one and the ones it
// was rotated from.
func ownObservers(st *store.Store, key ed25519.PrivateKey) ([][]byte, error) {
	out := [][]byte{key.Public().(ed25519.PublicKey)}
	book, err := st.TrustBook()
	if err != nil {
		return nil, err
	}
	for _, e := range book.Entries() {
		if e.Level == trust.Self {
			if k, err := hex.DecodeString(e.Key); err == nil && len(k) == ed25519.PublicKeySize && !bytes.Equal(k, out[0]) {
				out = append(out, k)
			}
		}
	}
	return out, nil
}

// pack writes observations as .rkm files that each fit a report.
func pack(list []*obs.Observation) ([][]byte, error) {
	var buf bytes.Buffer
	if err := rkm.Write(&buf, list, "rakkan-node"); err != nil {
		return nil, err
	}
	if buf.Len() <= mesh.MaxReport {
		return [][]byte{buf.Bytes()}, nil
	}
	if len(list) == 1 {
		return nil, errors.New("one observation does not fit a report")
	}
	a, err := pack(list[:len(list)/2])
	if err != nil {
		return nil, err
	}
	b, err := pack(list[len(list)/2:])
	return append(a, b...), err
}

func cmdPublish(args []string) error {
	var g globals
	var nf nodeFlags
	fs := flag.NewFlagSet("publish", flag.ExitOnError)
	g.register(fs)
	nf.register(fs)
	every := fs.Duration("every", 0, "keep publishing what is new at this interval (0: once)")
	location := fs.String("location", "keep", "positions: keep, drop, coarse:1km")
	wanFlag := fs.String("wan", "keep", "external addresses of your networks: keep, drop")
	all := fs.Bool("all", false, "report everything again, not only what is new")
	keep := fs.Duration("keep", 0, "also drop observations older than this from the store (0: no pruning; the daemon prunes by itself)")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), `usage: rakkan-node map publish [flags]

Hands this observer's observations to the local rakkan-node, which seals them to
the root key of the network: only the holder of that key can read them (neither
the nodes that carry them nor other members can). Your own store and what you
see in it stay yours; the network's map is the root's alone.

What is sent is everything you observed, including addresses, names, and
positions: --location drop and --wan drop keep those back.

`)
		fs.PrintDefaults()
	}
	fs.Parse(args)
	policy, err := policyFrom("own", "", *location, *wanFlag)
	if err != nil {
		return err
	}
	key, err := loadKey(g.key)
	if err != nil {
		return err
	}
	client, err := nf.client()
	if err != nil {
		return err
	}
	return loop(*every, func() error {
		if err := publishOnce(g, apiSender(client), key, policy, *all); err != nil {
			return err
		}
		if *keep > 0 {
			return pruneStore(g, *keep)
		}
		return nil
	})
}

// batch bounds how many observations one round reads at a time.
const batch = 2000

func publishOnce(g globals, send func([]byte) error, key ed25519.PrivateKey, policy share.Policy, all bool) error {
	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	if all {
		if err := st.ResetSent(); err != nil {
			return err
		}
	}
	own, err := ownObservers(st, key)
	if err != nil {
		return err
	}
	book, err := st.TrustBook()
	if err != nil {
		return err
	}
	total, reports := 0, 0
	var rep share.Report
	for {
		fresh, err := st.Unsent(own, batch)
		if err != nil {
			return err
		}
		if len(fresh) == 0 {
			break
		}
		shared, r, err := share.Apply(fresh, policy, key, book)
		if err != nil {
			return err
		}
		rep = r
		if len(shared) > 0 {
			files, err := pack(shared)
			if err != nil {
				return err
			}
			for _, f := range files {
				if err := send(f); err != nil {
					return err
				}
			}
			reports += len(files)
		}
		total += len(shared)
		if err := st.MarkSent(fresh); err != nil {
			return err
		}
	}
	if reports == 0 {
		fmt.Fprintln(os.Stderr, "nothing new to report")
		return nil
	}
	fmt.Fprintf(os.Stderr, "reported %d observations in %d sealed report(s); %s\n", total, reports, rep)
	return nil
}

// ---------- collect ----------

func cmdCollect(args []string) error {
	var nf nodeFlags
	fs := flag.NewFlagSet("collect", flag.ExitOnError)
	db := fs.String("db", collector.DefaultDB(), "the network store")
	nf.register(fs)
	rootKey := fs.String("root-key", "root.key", "the network's root key: the only thing that opens the reports")
	crlPath := fs.String("crl", "", "revocation list (crl.json): drop reports of revoked members")
	every := fs.Duration("every", 0, "keep collecting at this interval (0: once)")
	keep := fs.Duration("keep", DefaultKeep, "drop observations older than this from the network store (0: keep all)")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), `usage: rakkan-node map collect [--root-key root.key] [flags]

Only for the holder of the network's root key. Takes the sealed reports that
members published from a node (any node of the network: they all keep them for a
month), opens them with the root key, and merges the observations into the
network store (default `+collector.DefaultDB()+`).
Then look at the whole network:

    rakkan-node map serve --db `+collector.DefaultDB()+`
    rakkan-web --root-key root.key            (the page of the node, with the map)

The key is read here and never goes to the node. The network store is separate
from your own (the default --db of the other commands).

`)
		fs.PrintDefaults()
	}
	fs.Parse(args)
	root, err := collector.LoadRoot(*rootKey, *crlPath)
	if err != nil {
		return err
	}
	client, err := nf.client()
	if err != nil {
		return err
	}
	c := &collector.Collector{Root: root, Client: client, DB: *db, Keep: *keep}
	return loop(*every, func() error {
		stats, err := c.Once()
		fmt.Fprintln(os.Stderr, stats)
		return err
	})
}
