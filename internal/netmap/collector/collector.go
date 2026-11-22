// Package collector is the root's side of the network map: it takes the
// sealed reports members published from a node's API, opens them with the
// root key, and merges the observations into a store of the whole network.
// Only the holder of the root key can run it to any effect.
package collector

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/rkm"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
	"github.com/optimist3301/rakkan-mesh/internal/sealed"
)

// DefaultDB is where the network's map is kept, next to the observer's own.
func DefaultDB() string {
	return filepath.Join(filepath.Dir(store.DefaultPath()), "network.db")
}

// Root is the network and the key that opens its reports.
type Root struct {
	Network *membership.Network
	Key     ed25519.PrivateKey
	Revoked func(serial string) bool // nil: nobody
}

// LoadRoot reads the root key (a libp2p Ed25519 key file) and, if crlPath is
// not empty, the revocation list.
func LoadRoot(keyPath, crlPath string) (*Root, error) {
	sk, err := membership.LoadKey(keyPath)
	if err != nil {
		return nil, fmt.Errorf("root key: %w", err)
	}
	id, err := peer.IDFromPrivateKey(sk)
	if err != nil {
		return nil, err
	}
	nw, err := membership.ParseNetwork(id.String())
	if err != nil {
		return nil, err
	}
	raw, err := sk.Raw()
	if err != nil || len(raw) != ed25519.PrivateKeySize {
		return nil, errors.New("root key: not an Ed25519 key")
	}
	r := &Root{Network: nw, Key: ed25519.PrivateKey(raw)}
	if crlPath != "" {
		s, err := membership.LoadCRL(crlPath)
		if err != nil {
			return nil, err
		}
		rev := membership.NewRevocations(nw)
		if _, err := rev.Update(s); err != nil {
			return nil, err
		}
		r.Revoked = rev.Revoked
	}
	return r, nil
}

// Stats counts what one collection did.
type Stats struct{ Packets, Bad, Foreign, Observations, Added, Pruned, Guests int }

func (s Stats) String() string {
	guests := ""
	if s.Guests > 0 {
		guests = fmt.Sprintf(" (%d from guests)", s.Guests)
	}
	return fmt.Sprintf("%d report(s): %d observations%s, %d new; %d refused, %d not for this root, %d old ones dropped",
		s.Packets, s.Observations, guests, s.Added, s.Bad, s.Foreign, s.Pruned)
}

// Collector reads a node's spool into a store.
type Collector struct {
	Root   *Root
	Client *http.Client  // to the node's API socket (nodesock.Client)
	DB     string        // the network store (default DefaultDB)
	Keep   time.Duration // observations older than this are dropped after each run (0: keep all)
}

// Once collects what is new in the node's spool.
func (c *Collector) Once() (Stats, error) {
	var stats Stats
	db := c.DB
	if db == "" {
		db = DefaultDB()
	}
	var status struct {
		ID      string `json:"id"`
		Network string `json:"network"`
	}
	if err := getJSON(c.Client, "http://rakkan/v1/status", &status); err != nil {
		return stats, err
	}
	if status.Network != c.Root.Network.ID {
		return stats, fmt.Errorf("the node is in network %s, the root key is of %s", status.Network, c.Root.Network.ID)
	}
	st, err := store.Open(db)
	if err != nil {
		return stats, err
	}
	defer st.Close()
	cur, err := loadCursors(db + ".cursor")
	if err != nil {
		return stats, err
	}
	book, err := st.TrustBook()
	if err != nil {
		return stats, err
	}
	for {
		var page struct {
			Reports []struct {
				Seq    int64  `json:"seq"`
				Packet []byte `json:"packet"`
			} `json:"reports"`
		}
		if err := getJSON(c.Client, fmt.Sprintf("http://rakkan/v1/reports?after=%d", cur[status.ID]), &page); err != nil {
			return stats, err
		}
		if len(page.Reports) == 0 {
			break
		}
		for _, r := range page.Reports {
			stats.Packets++
			if book, err = c.open(st, book, r.Packet, &stats); err != nil {
				return stats, err
			}
			cur[status.ID] = r.Seq
		}
		if err := saveCursors(db+".cursor", cur); err != nil {
			return stats, err
		}
	}
	if c.Keep > 0 {
		if stats.Pruned, err = st.Prune(time.Now().Add(-c.Keep).UnixMilli(), nil); err != nil {
			return stats, err
		}
	}
	return stats, nil
}

func (c *Collector) open(st *store.Store, book *trust.Book, packet []byte, stats *Stats) (*trust.Book, error) {
	p, cert, err := mesh.VerifyReportGuests(c.Root.Network, packet, time.Now(), c.Root.Revoked, 0)
	if err != nil {
		stats.Bad++
		return book, nil
	}
	plain, err := sealed.Open(c.Root.Key, p.Sealed, mesh.ReportContext(c.Root.Network.ID))
	if err != nil {
		stats.Foreign++
		return book, nil
	}
	res, err := rkm.Read(bytes.NewReader(plain))
	if err != nil {
		stats.Bad++
		return book, nil
	}
	stats.Bad += len(res.Rejected)
	// A certificate is the root's own vouching for the member, so what a
	// member observed counts as trusted, unless the root decided otherwise.
	// A guest has only proved work: its observers are marked as guests, so
	// the map weighs them less and checks them against the members' word.
	guest := cert.HasRole(membership.RoleGuest)
	for _, o := range res.Observations {
		k := hex.EncodeToString(o.Observer)
		known, ok := book.Get(k)
		var e trust.Entry
		switch {
		case !ok && guest:
			e = trust.Entry{Key: k, Level: trust.Unknown, Guest: true, Name: "guest " + cert.Subject.ShortString()}
		case !ok:
			e = trust.Entry{Key: k, Level: trust.Trusted, Name: "member " + cert.Subject.ShortString()}
		case known.Guest && known.Level == trust.Unknown && !guest:
			// A member vouches for a key a guest brought first.
			e = trust.Entry{Key: k, Level: trust.Trusted, Name: "member " + cert.Subject.ShortString()}
		default:
			continue
		}
		if err := st.SetTrust(e); err != nil {
			return book, err
		}
		book = trust.NewBook(append(book.Entries(), e)...)
	}
	added, err := st.Add(res.Observations)
	if err != nil {
		return book, err
	}
	stats.Observations += len(res.Observations)
	if guest {
		stats.Guests += len(res.Observations)
	}
	stats.Added += added
	return book, nil
}

func getJSON(c *http.Client, url string, v any) error {
	res, err := c.Get(url)
	if err != nil {
		return fmt.Errorf("the node: %w", err)
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 64<<20))
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("the node answered %d: %s", res.StatusCode, bytes.TrimSpace(body))
	}
	return json.Unmarshal(body, v)
}

// cursors remember, per node, how far its spool has been read: the order of
// arrival is each node's own.
type cursors map[string]int64

func loadCursors(path string) (cursors, error) {
	c := cursors{}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	return c, json.Unmarshal(b, &c)
}

func saveCursors(path string, c cursors) error {
	b, _ := json.Marshal(c)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
