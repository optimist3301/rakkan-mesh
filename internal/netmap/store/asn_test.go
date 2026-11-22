package store

import (
	"bytes"
	"compress/gzip"
	"net/netip"
	"path/filepath"
	"strings"
	"testing"
)

const sample = "1.0.0.0\t1.0.0.255\t13335\tUS\tCLOUDFLARENET\n" +
	"1.0.1.0\t1.0.3.255\t0\tNone\tNot routed\n" +
	"81.2.69.0\t81.2.69.255\t20712\tGB\tANDREWS-ARNOLD\n" +
	"2001:4860::\t2001:4860:ffff:ffff:ffff:ffff:ffff:ffff\t15169\tUS\tGOOGLE\n"

func TestASN(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "m.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if st.ASNLookup() != nil {
		t.Fatal("lookup without a dataset must be nil")
	}
	var gz bytes.Buffer
	zw := gzip.NewWriter(&gz)
	zw.Write([]byte(sample))
	zw.Close()
	n, err := st.ImportASN(&gz)
	if err != nil || n != 3 {
		t.Fatalf("import: %d %v", n, err)
	}
	look := st.ASNLookup()
	for ip, want := range map[string]int{"1.0.0.1": 13335, "1.0.0.255": 13335, "81.2.69.160": 20712,
		"2001:4860:4860::8888": 15169, "1.0.2.1": 0, "1.0.4.0": 0, "10.0.0.1": 0, "::ffff:81.2.69.1": 20712} {
		as, ok := look(netip.MustParseAddr(ip))
		if (want == 0) == ok || (ok && as.Number != want) {
			t.Errorf("%s: got %+v %v, want AS%d", ip, as, ok, want)
		}
	}
	// Re-import replaces, never merges.
	if n, _ := st.ImportASN(strings.NewReader("8.8.8.0\t8.8.8.255\t15169\tUS\tGOOGLE\n")); n != 1 {
		t.Fatal("reimport")
	}
	if _, ok := st.ASNLookup()(netip.MustParseAddr("1.0.0.1")); ok {
		t.Fatal("old ranges survived a re-import")
	}
	if _, err := st.ImportASN(strings.NewReader("garbage line\n")); err == nil {
		t.Fatal("malformed dataset accepted")
	}
}

func TestASName(t *testing.T) {
	for in, want := range map[string]string{"AS20712 Andrews & Arnold Ltd": "Andrews & Arnold Ltd",
		"CLOUDFLARENET - Cloudflare, Inc.": "CLOUDFLARENET - Cloudflare, Inc.", "AS207123 Other": "AS207123 Other", "AS20712": "AS20712"} {
		if got := asName(20712, in); got != want {
			t.Errorf("%q → %q, want %q", in, got, want)
		}
	}
}
