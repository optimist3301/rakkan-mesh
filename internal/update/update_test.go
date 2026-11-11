package update

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestVersionOrder(t *testing.T) {
	order := []string{"0.009-alpha", "0.010-alpha", "0.010-beta", "0.010-rc1", "0.010", "0.011-alpha", "1.0-alpha", "1.0.1"}
	for i, a := range order {
		for j, b := range order {
			va, ok1 := ParseVersion(a)
			vb, ok2 := ParseVersion("v" + b)
			if !ok1 || !ok2 {
				t.Fatalf("cannot parse %s or %s", a, b)
			}
			want := 0
			if i < j {
				want = -1
			} else if i > j {
				want = 1
			}
			if got := va.Compare(vb); got != want {
				t.Errorf("%s vs %s = %d, want %d", a, b, got, want)
			}
		}
	}
	for _, bad := range []string{"dev", "", "abc-dirty"} {
		if _, ok := ParseVersion(bad); ok {
			t.Errorf("%q parsed", bad)
		}
	}
}

func fakeGitHub(t *testing.T, tags string, files map[string][]byte) *Updater {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/repos/o/r/releases", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, tags) })
	mux.HandleFunc("/o/r/releases/download/", func(w http.ResponseWriter, r *http.Request) {
		b, ok := files[strings.TrimPrefix(r.URL.Path, "/o/r/releases/download/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(b)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	// The programs in these archives are text, not something to run.
	accept := func(string, string) error { return nil }
	return &Updater{Repo: "o/r", APIBase: srv.URL, DownloadBase: srv.URL, GOOS: "linux", GOARCH: "amd64", Verify: accept}
}

func TestLatest(t *testing.T) {
	u := fakeGitHub(t, `[
	 {"tag_name":"v0.012-alpha","draft":true,"prerelease":true},
	 {"tag_name":"v0.011-alpha","prerelease":true},
	 {"tag_name":"v0.010-alpha","prerelease":true},
	 {"tag_name":"v0.009","prerelease":false},
	 {"tag_name":"nightly"}]`, nil)
	ctx := context.Background()
	r, err := u.Latest(ctx, "0.010-alpha")
	if err != nil || r == nil || r.Tag != "v0.011-alpha" || r.Version != "0.011-alpha" {
		t.Fatalf("got %+v, %v", r, err)
	}
	if r, err := u.Latest(ctx, "0.011-alpha"); err != nil || r != nil {
		t.Fatalf("up to date: got %+v, %v", r, err)
	}
	// A stable node does not move to a pre-release.
	if r, err := u.Latest(ctx, "0.008"); err != nil || r == nil || r.Tag != "v0.009" {
		t.Fatalf("stable: got %+v, %v", r, err)
	}
	if _, err := u.Latest(ctx, "dev"); err == nil {
		t.Fatal("a dev build was offered an update")
	}
}

func tarball(t *testing.T, name string, files map[string]string) []byte {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for n, c := range files {
		tw.WriteHeader(&tar.Header{Name: name + "/" + n, Mode: 0o755, Size: int64(len(c)), Typeflag: tar.TypeReg})
		tw.Write([]byte(c))
	}
	tw.Close()
	gz.Close()
	return buf.Bytes()
}

func sum(archive string, b []byte) []byte {
	h := sha256.Sum256(b)
	return []byte(hex.EncodeToString(h[:]) + "  " + archive + "\n")
}

func TestApply(t *testing.T) {
	name := "rakkan-node-0.011-alpha-linux-amd64"
	tgz := tarball(t, name, map[string]string{"rakkan-node": "NEW NODE", "rakkan-web": "NEW WEB", "README.md": "x"})
	dir := t.TempDir()
	exe := filepath.Join(dir, "rakkan-node")
	web := filepath.Join(dir, "rakkan-web")
	os.WriteFile(exe, []byte("old node"), 0o755)
	os.WriteFile(web, []byte("old web"), 0o755)

	u := fakeGitHub(t, "[]", map[string][]byte{
		"v0.011-alpha/" + name + ".tar.gz": tgz,
		"v0.011-alpha/SHA256SUMS":          sum(name+".tar.gz", tgz),
	})
	rel := &Release{Tag: "v0.011-alpha", Version: "0.011-alpha"}
	if err := u.Apply(context.Background(), rel, exe); err != nil {
		t.Fatal(err)
	}
	for p, want := range map[string]string{exe: "NEW NODE", web: "NEW WEB"} {
		if b, _ := os.ReadFile(p); string(b) != want {
			t.Errorf("%s = %q, want %q", p, b, want)
		}
	}
	// Windows has no executable bits.
	if st, _ := os.Stat(exe); runtime.GOOS != "windows" && st.Mode().Perm()&0o111 == 0 {
		t.Error("new program is not executable")
	}
	if left, _ := filepath.Glob(filepath.Join(dir, ".rakkan-update-*")); len(left) != 0 {
		t.Errorf("temp files left: %v", left)
	}
}

func TestApplyRejectsBadChecksum(t *testing.T) {
	name := "rakkan-node-0.011-alpha-linux-amd64"
	tgz := tarball(t, name, map[string]string{"rakkan-node": "NEW NODE"})
	dir := t.TempDir()
	exe := filepath.Join(dir, "rakkan-node")
	os.WriteFile(exe, []byte("old node"), 0o755)
	u := fakeGitHub(t, "[]", map[string][]byte{
		"v0.011-alpha/" + name + ".tar.gz": tgz,
		"v0.011-alpha/SHA256SUMS":          sum(name+".tar.gz", []byte("something else")),
	})
	err := u.Apply(context.Background(), &Release{Tag: "v0.011-alpha", Version: "0.011-alpha"}, exe)
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("err = %v", err)
	}
	if b, _ := os.ReadFile(exe); string(b) != "old node" {
		t.Errorf("program was replaced despite the bad checksum: %q", b)
	}
}

func TestApplyZip(t *testing.T) {
	name := "rakkan-node-0.011-alpha-windows-amd64"
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create(name + "/rakkan-node.exe")
	w.Write([]byte("NEW EXE"))
	zw.Close()
	dir := t.TempDir()
	exe := filepath.Join(dir, "rakkan-node.exe")
	os.WriteFile(exe, []byte("old"), 0o755)
	u := fakeGitHub(t, "[]", map[string][]byte{
		"v0.011-alpha/" + name + ".zip": buf.Bytes(),
		"v0.011-alpha/SHA256SUMS":       sum(name+".zip", buf.Bytes()),
	})
	u.GOOS = "windows"
	if err := u.Apply(context.Background(), &Release{Tag: "v0.011-alpha", Version: "0.011-alpha"}, exe); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(exe); string(b) != "NEW EXE" {
		t.Errorf("exe = %q", b)
	}
}

// A release whose program does not run here leaves the installed one alone.
func TestApplyKeepsNodeThatWouldNotRun(t *testing.T) {
	name := "rakkan-node-0.011-alpha-linux-amd64"
	tgz := tarball(t, name, map[string]string{"rakkan-node": "NEW NODE", "rakkan-web": "NEW WEB"})
	dir := t.TempDir()
	exe := filepath.Join(dir, "rakkan-node")
	web := filepath.Join(dir, "rakkan-web")
	os.WriteFile(exe, []byte("old node"), 0o755)
	os.WriteFile(web, []byte("old web"), 0o755)
	u := fakeGitHub(t, "[]", map[string][]byte{
		"v0.011-alpha/" + name + ".tar.gz": tgz,
		"v0.011-alpha/SHA256SUMS":          sum(name+".tar.gz", tgz),
	})
	u.Verify = RunsAs // the real check: "NEW NODE" is no program
	err := u.Apply(context.Background(), &Release{Tag: "v0.011-alpha", Version: "0.011-alpha"}, exe)
	if err == nil || !strings.Contains(err.Error(), "keeping the current version") {
		t.Fatalf("err = %v", err)
	}
	for p, want := range map[string]string{exe: "old node", web: "old web"} {
		if b, _ := os.ReadFile(p); string(b) != want {
			t.Errorf("%s = %q, want %q", p, b, want)
		}
	}
	if left, _ := filepath.Glob(filepath.Join(dir, ".rakkan-update-*")); len(left) != 0 {
		t.Errorf("temp files left: %v", left)
	}
}

func TestRunsAs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a shell script")
	}
	dir := t.TempDir()
	prog := filepath.Join(dir, "rakkan-node")
	os.WriteFile(prog, []byte("#!/bin/sh\necho rakkan-node 0.015-alpha linux/amd64 go1.26\n"), 0o755)
	if err := RunsAs(prog, "0.015-alpha"); err != nil {
		t.Fatal(err)
	}
	if err := RunsAs(prog, "0.016-alpha"); err == nil {
		t.Fatal("a program of another version passed")
	}
	os.WriteFile(prog, []byte("#!/bin/sh\necho 'version GLIBC_2.38 not found' >&2\nexit 1\n"), 0o755)
	if err := RunsAs(prog, "0.015-alpha"); err == nil || !strings.Contains(err.Error(), "GLIBC_2.38") {
		t.Fatalf("err = %v", err)
	}
}
