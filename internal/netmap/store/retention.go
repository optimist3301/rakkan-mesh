package store

import (
	"database/sql"
	"fmt"
	"strings"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// migrate adds what later versions need to a store made by an earlier one.
func migrate(db *sql.DB) error {
	var has int
	if err := db.QueryRow(`SELECT count(*) FROM pragma_table_info('obs') WHERE name = 'sent'`).Scan(&has); err != nil {
		return err
	}
	if has == 0 {
		// sent: the observation was handed to the network (reports to the root).
		if _, err := db.Exec(`ALTER TABLE obs ADD COLUMN sent INTEGER NOT NULL DEFAULT 0`); err != nil {
			return err
		}
	}
	if err := db.QueryRow(`SELECT count(*) FROM pragma_table_info('trust') WHERE name = 'guest'`).Scan(&has); err != nil {
		return err
	}
	if has == 0 {
		if _, err := db.Exec(`ALTER TABLE trust ADD COLUMN guest INTEGER NOT NULL DEFAULT 0`); err != nil {
			return err
		}
	}
	_, err := db.Exec(`CREATE INDEX IF NOT EXISTS obs_unsent ON obs(observer) WHERE sent = 0`)
	return err
}

// Unsent returns up to limit observations by the given observers that were
// not yet marked sent, oldest first. A node reporting every few minutes
// reads only what is new, however large the store is.
func (s *Store) Unsent(observers [][]byte, limit int) ([]*obs.Observation, error) {
	if len(observers) == 0 {
		return nil, nil
	}
	args := make([]any, 0, len(observers)+1)
	for _, o := range observers {
		args = append(args, o)
	}
	args = append(args, limit)
	q := `SELECT body, sig FROM obs WHERE sent = 0 AND observer IN (?` + strings.Repeat(",?", len(observers)-1) + `) ORDER BY t, id LIMIT ?`
	return s.query(q, args...)
}

func (s *Store) query(q string, args ...any) ([]*obs.Observation, error) {
	rows, err := s.db.Query(q, args...)
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

// MarkSent records that the observations went out.
func (s *Store) MarkSent(list []*obs.Observation) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	st, err := tx.Prepare(`UPDATE obs SET sent = 1 WHERE id = ?`)
	if err != nil {
		return err
	}
	defer st.Close()
	for _, o := range list {
		if _, err := st.Exec(o.ID[:]); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// ResetSent makes every observation unsent again.
func (s *Store) ResetSent() error {
	_, err := s.db.Exec(`UPDATE obs SET sent = 0`)
	return err
}

// Since loads the observations that started at or after cutoff (Unix ms),
// newest at most limit of them (0: all), in the order All uses. A viewer of
// a store that grows for months reads a window, not the history.
func (s *Store) Since(cutoff int64, limit int) ([]*obs.Observation, error) {
	if limit <= 0 {
		return s.query(`SELECT body, sig FROM obs WHERE t >= ? ORDER BY t, id`, cutoff)
	}
	return s.query(`SELECT body, sig FROM (SELECT t, id, body, sig FROM obs WHERE t >= ? ORDER BY t DESC, id DESC LIMIT ?) ORDER BY t, id`, cutoff, limit)
}

// Prune deletes observations that started before cutoff (Unix ms), except
// unsent ones by the observers in keep (own ones that have not reached the
// network yet), and returns how many went. It then shrinks the file: a store
// that only ever grows is a leak.
func (s *Store) Prune(cutoff int64, keep [][]byte) (int, error) {
	q := `DELETE FROM obs WHERE t < ?`
	args := []any{cutoff}
	if len(keep) > 0 {
		q += ` AND NOT (sent = 0 AND observer IN (?` + strings.Repeat(",?", len(keep)-1) + `))`
		for _, k := range keep {
			args = append(args, k)
		}
	}
	r, err := s.db.Exec(q, args...)
	if err != nil {
		return 0, err
	}
	n, _ := r.RowsAffected()
	if n > 0 {
		// The database file keeps its pages until they are vacuumed, and
		// the write-ahead log until it is truncated.
		if _, err := s.db.Exec(`VACUUM`); err != nil {
			return int(n), err
		}
		s.db.Exec(`PRAGMA wal_checkpoint(TRUNCATE)`)
	}
	return int(n), nil
}

// Version names the state of the observation set for caches: it changes
// whenever observations are added or pruned.
func (s *Store) Version() (string, error) {
	var n, mx, mn int64
	if err := s.db.QueryRow(`SELECT count(*), coalesce(max(t),0), coalesce(min(t),0) FROM obs`).Scan(&n, &mx, &mn); err != nil {
		return "", err
	}
	return fmt.Sprintf("%d/%d/%d", n, mn, mx), nil
}
