package web

import (
	"os/exec"
	"path/filepath"
	"testing"
)

// The search language lives in static/search.js; its tests run in node when
// it is installed.
func TestSearchJS(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed")
	}
	lib, _ := filepath.Abs("static/search.js")
	out, err := exec.Command(node, "testdata/search_test.js", lib).CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
}
