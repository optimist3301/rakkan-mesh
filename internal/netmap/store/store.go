// Package store is the local working database: an append-only set of
// observations in SQLite. Inserting is idempotent, so importing any map any
// number of times in any order yields the same set (spec §1).
package store

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

type Store struct{ db *sql.DB }

func DefaultPath() string {
	dir := os.Getenv("XDG_DATA_HOME")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(dir, "rakkan", "map.db")
}

const schema = `
CREATE TABLE IF NOT EXISTS obs (
	id       BLOB PRIMARY KEY,
	observer BLOB NOT NULL,
	t        INTEGER NOT NULL,
	mode     INTEGER NOT NULL,
	method   TEXT NOT NULL,
	body     BLOB NOT NULL,
	sig      BLOB NOT NULL
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS obs_t ON obs(t);
CREATE INDEX IF NOT EXISTS obs_observer ON obs(observer);
CREATE TABLE IF NOT EXISTS trust (
	key   TEXT PRIMARY KEY,  -- hex ed25519 public key
	level INTEGER NOT NULL,
	name  TEXT NOT NULL DEFAULT '',
	guest INTEGER NOT NULL DEFAULT 0  -- reached us only through a guest
);
`

func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schema + asnSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}
	if err := migrate(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

// Add inserts observations; returns how many were new.
func (s *Store) Add(list []*obs.Observation) (int, error) {
	tx, err := s.db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	st, err := tx.Prepare(`INSERT OR IGNORE INTO obs(id, observer, t, mode, method, body, sig) VALUES (?,?,?,?,?,?,?)`)
	if err != nil {
		return 0, err
	}
	defer st.Close()
	added := 0
	for _, o := range list {
		r, err := st.Exec(o.ID[:], o.Observer, o.Time, int(o.Mode), o.Method, o.Raw, o.Sig)
		if err != nil {
			return 0, err
		}
		n, _ := r.RowsAffected()
		added += int(n)
	}
	return added, tx.Commit()
}

// All loads every stored observation. Signatures were verified before the
// observations were added (Add only accepts Sign/Decode results), so they
// are not re-checked here; Verify does that on demand.
func (s *Store) All() ([]*obs.Observation, error) {
	rows, err := s.db.Query(`SELECT body, sig FROM obs ORDER BY t, id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*obs.Observation
	for rows.Next() {
		var body, sig []byte
		if err := rows.Scan(&body, &sig); err != nil {
			return nil, err
		}
		o, err := obs.DecodeStored(body, sig)
		if err != nil {
			return nil, fmt.Errorf("corrupt observation in store: %w", err)
		}
		out = append(out, o)
	}
	return out, rows.Err()
}

// Verify re-checks every signature and canonical encoding in the store.
func (s *Store) Verify() (ok, bad int, err error) {
	rows, err := s.db.Query(`SELECT body, sig FROM obs`)
	if err != nil {
		return 0, 0, err
	}
	defer rows.Close()
	for rows.Next() {
		var body, sig []byte
		if err := rows.Scan(&body, &sig); err != nil {
			return ok, bad, err
		}
		if _, err := obs.Decode(body, sig); err != nil {
			bad++
		} else {
			ok++
		}
	}
	return ok, bad, rows.Err()
}

type Stats struct {
	Observations int
	Observers    int
	ByMode       map[obs.Mode]int
}

func (s *Store) Stats() (Stats, error) {
	st := Stats{ByMode: map[obs.Mode]int{}}
	if err := s.db.QueryRow(`SELECT count(*), count(DISTINCT observer) FROM obs`).Scan(&st.Observations, &st.Observers); err != nil {
		return st, err
	}
	rows, err := s.db.Query(`SELECT mode, count(*) FROM obs GROUP BY mode`)
	if err != nil {
		return st, err
	}
	defer rows.Close()
	for rows.Next() {
		var m, n int
		if err := rows.Scan(&m, &n); err != nil {
			return st, err
		}
		st.ByMode[obs.Mode(m)] = n
	}
	return st, rows.Err()
}

// TrustBook loads the local trust book (not part of any map).
func (s *Store) TrustBook() (*trust.Book, error) {
	rows, err := s.db.Query(`SELECT key, level, name, guest FROM trust`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var es []trust.Entry
	for rows.Next() {
		var e trust.Entry
		if err := rows.Scan(&e.Key, &e.Level, &e.Name, &e.Guest); err != nil {
			return nil, err
		}
		es = append(es, e)
	}
	return trust.NewBook(es...), rows.Err()
}

func (s *Store) SetTrust(e trust.Entry) error {
	_, err := s.db.Exec(`INSERT INTO trust(key, level, name, guest) VALUES (?,?,?,?)
		ON CONFLICT(key) DO UPDATE SET level=excluded.level, guest=excluded.guest,
			name=CASE WHEN excluded.name='' THEN trust.name ELSE excluded.name END`,
		e.Key, int(e.Level), e.Name, e.Guest)
	return err
}

func (s *Store) DeleteTrust(key string) error {
	_, err := s.db.Exec(`DELETE FROM trust WHERE key = ?`, key)
	return err
}

// Observers lists the hex keys of every observer in the store.
func (s *Store) Observers() ([]string, error) {
	rows, err := s.db.Query(`SELECT DISTINCT lower(hex(observer)) FROM obs`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var k string
		if err := rows.Scan(&k); err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, rows.Err()
}
