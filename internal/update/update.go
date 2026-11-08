// Package update finds newer rakkan-node releases on GitHub and installs
// them over the running program.
//
// A release is a tag such as v0.010-alpha with, for every platform, an
// archive rakkan-node-<version>-<os>-<arch>.tar.gz (.zip on Windows) and a
// SHA256SUMS file listing their checksums: what "make dist" builds. The
// archive is checked against SHA256SUMS before anything is replaced. Both
// come from the same release, so the check catches corrupt or truncated
// downloads, not a compromised release: the release is trusted as much as
// the repository it is published in.
package update

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// DefaultRepo is the GitHub repository releases are taken from.
const DefaultRepo = "optimist3301/rakkan-mesh"

const (
	maxAPIBytes     = 4 << 20
	maxSumsBytes    = 1 << 20
	maxArchiveBytes = 1 << 30
	maxBinaryBytes  = 1 << 30
)

// Version is a parsed release version such as 0.010-alpha.
type Version struct {
	Nums [3]int
	Pre  string // "alpha", "beta", "rc1"; empty for a stable release
}

var versionRE = regexp.MustCompile(`^v?(\d+)\.(\d+)(?:\.(\d+))?(?:-(.+))?$`)

// ParseVersion parses a release tag or version. Builds that are not
// releases, such as "dev", do not parse.
func ParseVersion(s string) (Version, bool) {
	m := versionRE.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return Version{}, false
	}
	var v Version
	for i := 0; i < 3; i++ {
		if m[i+1] == "" {
			continue
		}
		n, err := strconv.Atoi(m[i+1])
		if err != nil {
			return Version{}, false
		}
		v.Nums[i] = n
	}
	v.Pre = m[4]
	return v, true
}

// Compare returns -1, 0 or 1 as v is older than, equal to or newer than w.
// A stable release is newer than a pre-release of the same numbers.
func (v Version) Compare(w Version) int {
	for i := range v.Nums {
		if v.Nums[i] != w.Nums[i] {
			if v.Nums[i] < w.Nums[i] {
				return -1
			}
			return 1
		}
	}
	switch {
	case v.Pre == w.Pre:
		return 0
	case v.Pre == "":
		return 1
	case w.Pre == "":
		return -1
	case v.Pre < w.Pre:
		return -1
	}
	return 1
}

// Release is a published release.
type Release struct {
	Tag     string // v0.010-alpha
	Version string // 0.010-alpha, as in the archive names
	parsed  Version
}

// Updater talks to GitHub. The zero value, with Repo set, is ready to use.
type Updater struct {
	Repo   string       // owner/name; default DefaultRepo
	Client *http.Client // default: a client with a generous timeout

	// Overridable for tests.
	APIBase      string // default https://api.github.com
	DownloadBase string // default https://github.com
	GOOS, GOARCH string // default: this program's
}

func (u *Updater) repo() string {
	if u.Repo != "" {
		return u.Repo
	}
	return DefaultRepo
}

func (u *Updater) client() *http.Client {
	if u.Client != nil {
		return u.Client
	}
	return &http.Client{Timeout: 30 * time.Minute}
}

func or(s, def string) string {
	if s != "" {
		return s
	}
	return def
}

func (u *Updater) get(ctx context.Context, url string, limit int64, w io.Writer) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "rakkan-node-updater")
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := u.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	n, err := io.Copy(w, io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return err
	}
	if n > limit {
		return fmt.Errorf("%s: larger than %d bytes", url, limit)
	}
	return nil
}

// Latest returns the newest release that is newer than current, or nil when
// current is up to date. Pre-releases are considered only when current is
// one itself, so a node on a stable release stays on stable ones.
func (u *Updater) Latest(ctx context.Context, current string) (*Release, error) {
	cur, ok := ParseVersion(current)
	if !ok {
		return nil, fmt.Errorf("version %q is not a release: development builds are not updated", current)
	}
	var body limitedBuffer
	if err := u.get(ctx, or(u.APIBase, "https://api.github.com")+"/repos/"+u.repo()+"/releases?per_page=50", maxAPIBytes, &body); err != nil {
		return nil, err
	}
	var list []struct {
		Tag        string `json:"tag_name"`
		Draft      bool   `json:"draft"`
		Prerelease bool   `json:"prerelease"`
	}
	if err := json.Unmarshal(body.b, &list); err != nil {
		return nil, fmt.Errorf("releases: %w", err)
	}
	var best *Release
	for _, r := range list {
		v, ok := ParseVersion(r.Tag)
		if !ok || r.Draft || (r.Prerelease || v.Pre != "") && cur.Pre == "" {
			continue
		}
		if v.Compare(cur) <= 0 || best != nil && v.Compare(best.parsed) <= 0 {
			continue
		}
		best = &Release{Tag: r.Tag, Version: strings.TrimPrefix(r.Tag, "v"), parsed: v}
	}
	return best, nil
}

type limitedBuffer struct{ b []byte }

func (l *limitedBuffer) Write(p []byte) (int, error) { l.b = append(l.b, p...); return len(p), nil }

// Apply downloads the release for this platform, checks it, and replaces
// the program at exe, and rakkan-web beside it if there is one. The running
// process is not touched: it goes on with the old code until restarted.
func (u *Updater) Apply(ctx context.Context, rel *Release, exe string) error {
	goos, goarch := or(u.GOOS, runtime.GOOS), or(u.GOARCH, runtime.GOARCH)
	name := fmt.Sprintf("rakkan-node-%s-%s-%s", rel.Version, goos, goarch)
	archive := name + ".tar.gz"
	if goos == "windows" {
		archive = name + ".zip"
	}
	base := or(u.DownloadBase, "https://github.com") + "/" + u.repo() + "/releases/download/" + rel.Tag + "/"

	var sums limitedBuffer
	if err := u.get(ctx, base+"SHA256SUMS", maxSumsBytes, &sums); err != nil {
		return err
	}
	want := ""
	for _, line := range strings.Split(string(sums.b), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && strings.TrimPrefix(f[1], "*") == archive {
			want = strings.ToLower(f[0])
		}
	}
	if want == "" {
		return fmt.Errorf("%s is not listed in the release's SHA256SUMS", archive)
	}

	dir := filepath.Dir(exe)
	tmp, err := os.MkdirTemp(dir, ".rakkan-update-")
	if err != nil {
		return fmt.Errorf("cannot write next to %s: %w", exe, err)
	}
	defer os.RemoveAll(tmp)
	af, err := os.Create(filepath.Join(tmp, archive))
	if err != nil {
		return err
	}
	defer af.Close()
	h := sha256.New()
	if err := u.get(ctx, base+archive, maxArchiveBytes, io.MultiWriter(af, h)); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("checksum mismatch for %s (expected %s, got %s)", archive, want, got)
	}

	ext := ""
	if goos == "windows" {
		ext = ".exe"
	}
	// Programs to replace: the one running, and rakkan-web if installed beside it.
	targets := map[string]string{"rakkan-node" + ext: exe}
	if web := filepath.Join(dir, "rakkan-web"+ext); exists(web) {
		targets["rakkan-web"+ext] = web
	}
	var extracted map[string]string
	if goos == "windows" {
		extracted, err = extractZip(af, name, targets, tmp)
	} else {
		extracted, err = extractTar(af, name, targets, tmp)
	}
	if err != nil {
		return err
	}
	if _, ok := extracted["rakkan-node"+ext]; !ok {
		return fmt.Errorf("%s does not contain rakkan-node%s", archive, ext)
	}
	// The node first, so a failure on the web page's program leaves a
	// consistent node.
	for _, n := range []string{"rakkan-node" + ext, "rakkan-web" + ext} {
		src, ok := extracted[n]
		if !ok {
			continue
		}
		if err := replace(targets[n], src); err != nil {
			return err
		}
	}
	return nil
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

// extractTar unpacks the wanted programs from name/<program> in a .tar.gz.
func extractTar(f *os.File, name string, want map[string]string, tmp string) (map[string]string, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	tr := tar.NewReader(gz)
	out := map[string]string{}
	for {
		hd, err := tr.Next()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, err
		}
		base := path.Base(hd.Name)
		if hd.Typeflag != tar.TypeReg || path.Clean(hd.Name) != name+"/"+base {
			continue
		}
		if _, ok := want[base]; !ok {
			continue
		}
		if out[base], err = save(tr, filepath.Join(tmp, base+".new")); err != nil {
			return nil, err
		}
	}
}

// extractZip is extractTar for a .zip.
func extractZip(f *os.File, name string, want map[string]string, tmp string) (map[string]string, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	zr, err := zip.NewReader(f, st.Size())
	if err != nil {
		return nil, err
	}
	out := map[string]string{}
	for _, zf := range zr.File {
		base := path.Base(zf.Name)
		if zf.FileInfo().IsDir() || path.Clean(zf.Name) != name+"/"+base {
			continue
		}
		if _, ok := want[base]; !ok {
			continue
		}
		rc, err := zf.Open()
		if err != nil {
			return nil, err
		}
		out[base], err = save(rc, filepath.Join(tmp, base+".new"))
		rc.Close()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func save(r io.Reader, dst string) (string, error) {
	f, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o755)
	if err != nil {
		return "", err
	}
	n, err := io.Copy(f, io.LimitReader(r, maxBinaryBytes+1))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil && n > maxBinaryBytes {
		err = errors.New("program in the archive is too large")
	}
	if err != nil {
		return "", err
	}
	return dst, nil
}

// replace puts src in place of dst. On Unix a rename does it, even while
// dst runs. Windows will not overwrite a running program but will rename
// it, so the old one is moved aside (CleanUp removes it later).
func replace(dst, src string) error {
	if st, err := os.Stat(dst); err == nil {
		if err := os.Chmod(src, st.Mode().Perm()|0o111); err != nil {
			return err
		}
	}
	// The temp directory is inside dst's, so the rename stays on one volume.
	if runtime.GOOS != "windows" {
		return os.Rename(src, dst)
	}
	old := dst + ".old"
	os.Remove(old)
	if err := os.Rename(dst, old); err != nil {
		return err
	}
	if err := os.Rename(src, dst); err != nil {
		os.Rename(old, dst)
		return err
	}
	return nil
}

// CleanUp removes what a Windows update left of the previous program.
func CleanUp(exe string) {
	os.Remove(exe + ".old")
	ext := filepath.Ext(exe)
	os.Remove(filepath.Join(filepath.Dir(exe), "rakkan-web"+ext+".old"))
}
