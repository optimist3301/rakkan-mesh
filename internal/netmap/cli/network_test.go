//go:build linux

package cli

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/api"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/collector"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
)

// nodeAPI runs a real member node of the network with root's certificate and
// serves its report API, as rakkan-node does on its socket.
func nodeAPI(t *testing.T, root crypto.PrivKey, nw *membership.Network) *http.Client {
	t.Helper()
	sk, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	id, _ := peer.IDFromPrivateKey(sk)
	cert, err := membership.Issue(root, nil, membership.Cert{Network: nw.ID, Subject: id, Roles: []string{membership.RoleMember},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour)})
	if err != nil {
		t.Fatal(err)
	}
	node, err := mesh.New(mesh.Config{Key: sk, Network: nw, Cert: cert, ListenAddrs: []string{"/ip4/127.0.0.1/udp/0/quic-v1"},
		Reports: mesh.ReportsConfig{Dir: t.TempDir()}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { node.Close() })
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"id": id.String(), "network": nw.ID})
	})
	mux.Handle("/", (&api.Server{Node: node, Network: nw.ID}).Handler())
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, "tcp", srv.Listener.Addr().String())
	}}}
}

func collectWith(db string, client *http.Client, nw *membership.Network, key ed25519.PrivateKey) error {
	_, err := (&collector.Collector{Root: &collector.Root{Network: nw, Key: key}, Client: client, DB: db}).Once()
	return err
}

func observed(t *testing.T, dir string, key ed25519.PrivateKey, ips ...string) globals {
	t.Helper()
	g := globals{db: filepath.Join(dir, "map.db")}
	st, err := store.Open(g.db)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	var list []*obs.Observation
	for _, ip := range ips {
		o, err := obs.Sign(obs.Body{Time: time.Now().UnixMilli(), Mode: obs.ModeActive, Method: "ping",
			Subject: []string{"ip:" + ip}, Claims: []obs.Claim{{K: "name", V: "host-" + ip}}}, key)
		if err != nil {
			t.Fatal(err)
		}
		list = append(list, o)
	}
	if _, err := st.Add(list); err != nil {
		t.Fatal(err)
	}
	return g
}

func names(t *testing.T, db string) []string {
	t.Helper()
	st, err := store.Open(db)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	all, _ := st.All()
	var out []string
	for _, o := range all {
		out = append(out, strings.Join(o.Subject, ","))
	}
	return out
}

func TestOnlyTheRootSeesTheNetwork(t *testing.T) {
	rootSK, nw, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	rootRaw, _ := rootSK.Raw()
	root := ed25519.PrivateKey(rootRaw)
	client := nodeAPI(t, rootSK, nw)

	// Two members observe their own networks and publish.
	policy, _ := policyFrom("own", "", "keep", "keep")
	_, keyA, _ := ed25519.GenerateKey(rand.Reader)
	_, keyB, _ := ed25519.GenerateKey(rand.Reader)
	gA := observed(t, t.TempDir(), keyA, "10.0.0.5", "10.0.0.6")
	gB := observed(t, t.TempDir(), keyB, "192.168.1.9")
	for _, m := range []struct {
		g   globals
		key ed25519.PrivateKey
	}{{gA, keyA}, {gB, keyB}} {
		if err := publishOnce(m.g, apiSender(client), m.key, policy, false); err != nil {
			t.Fatal(err)
		}
	}
	// Reported once: a second run has nothing new.
	if err := publishOnce(gA, apiSender(client), keyA, policy, false); err != nil {
		t.Fatal(err)
	}

	// A member's own map holds only what it saw.
	if got := names(t, gA.db); len(got) != 2 {
		t.Fatalf("member A sees %v", got)
	}

	// The node holds only sealed bytes: no member's name or address is in them.
	var page struct{ Reports []struct{ Packet []byte } }
	res, err := client.Get("http://rakkan/v1/reports")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewDecoder(res.Body).Decode(&page); err != nil || len(page.Reports) != 2 {
		t.Fatalf("spool: %v, %d reports", err, len(page.Reports))
	}
	res.Body.Close()
	for _, r := range page.Reports {
		if strings.Contains(string(r.Packet), "10.0.0") || strings.Contains(string(r.Packet), "host-") {
			t.Fatal("a report is readable in the spool")
		}
	}

	// The root collects the whole network into a store of its own.
	netDB := filepath.Join(t.TempDir(), "network.db")
	collect := func(db string, k ed25519.PrivateKey, n *membership.Network) error {
		return collectWith(db, client, n, k)
	}
	if err := collect(netDB, root, nw); err != nil {
		t.Fatal(err)
	}
	if got := names(t, netDB); len(got) != 3 {
		t.Fatalf("the root sees %v", got)
	}
	st, _ := store.Open(netDB)
	book, _ := st.TrustBook()
	st.Close()
	if book.Level(hex.EncodeToString(keyA.Public().(ed25519.PublicKey))).String() != "trusted" {
		t.Fatalf("members' observations are not trusted: %v", book.Entries())
	}
	// Collecting again adds nothing and re-reads nothing.
	if err := collect(netDB, root, nw); err != nil {
		t.Fatal(err)
	}
	if got := names(t, netDB); len(got) != 3 {
		t.Fatalf("after a second collect: %v", got)
	}

	// A member's key (or anyone else's) opens none of it.
	_, otherRoot, _ := ed25519.GenerateKey(rand.Reader)
	wrong := &collector.Collector{Root: &collector.Root{Network: nw, Key: otherRoot}, Client: client, DB: filepath.Join(t.TempDir(), "x.db")}
	if stats, err := wrong.Once(); err != nil || stats.Foreign != 2 || stats.Added != 0 {
		t.Fatalf("a wrong key: %v %+v", err, stats)
	}

	// Another network's root is refused before anything is read.
	_, otherNW, _ := membership.NewRootKey()
	if err := collect(filepath.Join(t.TempDir(), "y.db"), root, otherNW); err == nil {
		t.Fatal("collected from a node of another network")
	}
}

func TestPublishKeepsBackWhatYouAskFor(t *testing.T) {
	rootSK, nw, _ := membership.NewRootKey()
	rootRaw, _ := rootSK.Raw()
	client := nodeAPI(t, rootSK, nw)
	_, key, _ := ed25519.GenerateKey(rand.Reader)
	g := globals{db: filepath.Join(t.TempDir(), "map.db")}
	st, _ := store.Open(g.db)
	here, _ := obs.Sign(obs.Body{Time: time.Now().UnixMilli(), Mode: obs.ModeLocal, Method: "self",
		Subject: []string{"ip:10.0.0.1"}, Claims: []obs.Claim{{K: "loc.geo", V: obs.FormatGeo(50.087, 14.42, 20)}, {K: "loc.logical", V: "Lab / desk 3"}, {K: "name", V: "laptop"}}}, key)
	st.Add([]*obs.Observation{here})
	st.Close()
	policy, err := policyFrom("own", "", "drop", "keep")
	if err != nil {
		t.Fatal(err)
	}
	if err := publishOnce(g, apiSender(client), key, policy, false); err != nil {
		t.Fatal(err)
	}
	netDB := filepath.Join(t.TempDir(), "n.db")
	if err := collectWith(netDB, client, nw, ed25519.PrivateKey(rootRaw)); err != nil {
		t.Fatal(err)
	}
	st, _ = store.Open(netDB)
	defer st.Close()
	all, _ := st.All()
	for _, o := range all {
		for _, c := range o.Claims {
			if strings.HasPrefix(c.K, "loc.") {
				t.Fatalf("a position was reported: %v", c)
			}
		}
	}
	if len(all) != 1 || len(all[0].Claims) != 1 {
		t.Fatalf("the rest should still be reported: %v", all)
	}
}

// A node reporting for months must not grow: the store is pruned, rounds
// read only what is new, and no goroutines or memory pile up.
func TestReporterRoundsStayBounded(t *testing.T) {
	_, key, _ := ed25519.GenerateKey(rand.Reader)
	dir := t.TempDir()
	keyPath := filepath.Join(dir, "observer.key")
	if err := obs.SaveKey(keyPath, key); err != nil {
		t.Fatal(err)
	}
	g := globals{db: filepath.Join(dir, "map.db"), key: keyPath}
	policy, _ := policyFrom("own", "", "keep", "keep")
	sent := 0
	send := func(b []byte) error { sent++; return nil }

	round := func(i int) {
		st, err := store.Open(g.db)
		if err != nil {
			t.Fatal(err)
		}
		var list []*obs.Observation
		for j := range 50 {
			o, err := obs.Sign(obs.Body{Time: time.Now().UnixMilli(), Mode: obs.ModeActive, Method: "ping",
				Subject: []string{"ip:10.0.0." + strconv.Itoa(j)}, Claims: []obs.Claim{{K: "round", V: strconv.Itoa(i)}}}, key)
			if err != nil {
				t.Fatal(err)
			}
			list = append(list, o)
		}
		st.Add(list)
		st.Close()
		if err := publishOnce(g, send, key, policy, false); err != nil {
			t.Fatal(err)
		}
		if err := pruneStore(g, 200*time.Millisecond); err != nil {
			t.Fatal(err)
		}
	}
	for i := range 5 {
		round(i)
	}
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	goroutines := runtime.NumGoroutine()
	for i := 5; i < 120; i++ {
		round(i)
		time.Sleep(15 * time.Millisecond)
	}
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	st, _ := store.Open(g.db)
	defer st.Close()
	if s, _ := st.Stats(); s.Observations > 50*20 {
		t.Fatalf("the store holds %d observations after 120 rounds of 50", s.Observations)
	}
	if got, _ := st.Unsent([][]byte{key.Public().(ed25519.PublicKey)}, 1); len(got) != 0 {
		t.Fatal("something was left unsent")
	}
	if sent != 120 {
		t.Fatalf("%d reports for 120 rounds", sent)
	}
	if grew := int64(after.HeapAlloc) - int64(before.HeapAlloc); grew > 8<<20 {
		t.Fatalf("heap grew by %d bytes over 115 rounds", grew)
	}
	if g2 := runtime.NumGoroutine(); g2 > goroutines+2 {
		t.Fatalf("goroutines %d -> %d", goroutines, g2)
	}
	if fi, err := os.Stat(g.db); err != nil || fi.Size() > 2<<20 {
		t.Fatalf("the database file is %v bytes: %v", fi.Size(), err)
	}
	if fi, err := os.Stat(g.db + "-wal"); err == nil && fi.Size() > 4<<20 {
		t.Fatalf("the write-ahead log is %d bytes", fi.Size())
	}
}
