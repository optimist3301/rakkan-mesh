package store

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"net/netip"
	"strconv"
	"strings"
	"sync"
)

// AS is an autonomous system: who announces an address on the internet.
type AS struct {
	Number  int    `json:"asn"`
	Name    string `json:"name"`
	Country string `json:"country,omitempty"` // where the AS is registered, not where the address is
}

const asnSchema = `
CREATE TABLE IF NOT EXISTS asn (
	start BLOB PRIMARY KEY,  -- 16-byte address (IPv4 mapped), big endian: byte order = address order
	stop  BLOB NOT NULL,
	asn   INTEGER NOT NULL,
	cc    TEXT NOT NULL,
	name  TEXT NOT NULL
) WITHOUT ROWID;
`

// ImportASN replaces the IP→AS table with a dataset in the ip2asn TSV
// format (range_start, range_end, AS_number, country_code, AS_description),
// plain or gzip-compressed. Reference data: it is local and never part of
// a map. Returns the number of routed ranges stored.
func (s *Store) ImportASN(r io.Reader) (int, error) {
	br := bufio.NewReader(r)
	if magic, _ := br.Peek(2); bytes.Equal(magic, []byte{0x1f, 0x8b}) {
		zr, err := gzip.NewReader(br)
		if err != nil {
			return 0, err
		}
		defer zr.Close()
		br = bufio.NewReader(zr)
	}
	if _, err := s.db.Exec(asnSchema); err != nil {
		return 0, err
	}
	tx, err := s.db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`DELETE FROM asn`); err != nil {
		return 0, err
	}
	st, err := tx.Prepare(`INSERT OR REPLACE INTO asn(start, stop, asn, cc, name) VALUES (?,?,?,?,?)`)
	if err != nil {
		return 0, err
	}
	defer st.Close()
	sc := bufio.NewScanner(br)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	n, line := 0, 0
	for sc.Scan() {
		line++
		f := strings.Split(sc.Text(), "\t")
		if len(f) < 5 {
			if strings.TrimSpace(sc.Text()) == "" {
				continue
			}
			return n, fmt.Errorf("line %d: want 5 tab-separated fields (ip2asn format)", line)
		}
		lo, err1 := netip.ParseAddr(f[0])
		hi, err2 := netip.ParseAddr(f[1])
		asn, err3 := strconv.Atoi(f[2])
		if err1 != nil || err2 != nil || err3 != nil || lo.Is4() != hi.Is4() || hi.Less(lo) {
			return n, fmt.Errorf("line %d: malformed range", line)
		}
		if asn == 0 {
			continue // not routed
		}
		a, b := lo.As16(), hi.As16()
		if _, err := st.Exec(a[:], b[:], asn, f[3], asName(asn, f[4])); err != nil {
			return n, err
		}
		n++
	}
	if err := sc.Err(); err != nil {
		return n, err
	}
	if n == 0 {
		return 0, errors.New("no routed ranges found")
	}
	return n, tx.Commit()
}

// asName drops a leading "AS<number>" that some descriptions repeat:
// "AS20712 Andrews & Arnold Ltd" → "Andrews & Arnold Ltd".
func asName(asn int, desc string) string {
	d := strings.TrimSpace(desc)
	if rest, ok := strings.CutPrefix(d, "AS"+strconv.Itoa(asn)); ok && (rest == "" || rest[0] == ' ' || rest[0] == '-') {
		if r := strings.TrimLeft(rest, " -"); r != "" {
			return r
		}
	}
	return d
}

// ASNLookup returns a cached lookup function, or nil if no dataset was imported.
func (s *Store) ASNLookup() func(netip.Addr) (AS, bool) {
	var count int
	if err := s.db.QueryRow(`SELECT count(*) FROM asn`).Scan(&count); err != nil || count == 0 {
		return nil
	}
	var mu sync.Mutex
	cache := map[netip.Addr]*AS{}
	return func(a netip.Addr) (AS, bool) {
		a = a.Unmap()
		mu.Lock()
		defer mu.Unlock()
		if c, ok := cache[a]; ok {
			if c == nil {
				return AS{}, false
			}
			return *c, true
		}
		k := a.As16()
		var stop []byte
		var as AS
		err := s.db.QueryRow(`SELECT stop, asn, cc, name FROM asn WHERE start <= ? ORDER BY start DESC LIMIT 1`, k[:]).
			Scan(&stop, &as.Number, &as.Country, &as.Name)
		if err != nil || bytes.Compare(k[:], stop) > 0 {
			if err != nil && !errors.Is(err, sql.ErrNoRows) {
				return AS{}, false // transient: do not cache
			}
			cache[a] = nil
			return AS{}, false
		}
		cache[a] = &as
		return as, true
	}
}
