package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestConfigApply(t *testing.T) {
	p := filepath.Join(t.TempDir(), "node.json")
	os.WriteFile(p, []byte(`{"workers":3,"status":"2s","mdns":false,"bootstrap":["a","b"],"private_key":null}`), 0o600)
	// null key is not a string, so it must be rejected.
	if _, err := loadConfig(p); err == nil {
		t.Fatal("null private_key accepted")
	}
	os.WriteFile(p, []byte(`{"workers":3,"status":"2s","mdns":false,"bootstrap":["a","b"]}`), 0o600)
	c, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	workers := fs.Int("workers", 1, "")
	status := fs.Duration("status", time.Second, "")
	mdns := fs.Bool("mdns", true, "")
	boot := fs.String("bootstrap", "", "")
	fs.Parse([]string{"-workers", "9"})
	if err := c.apply(fs); err != nil {
		t.Fatal(err)
	}
	if *workers != 9 || *status != 2*time.Second || *mdns || *boot != "a,b" {
		t.Fatalf("got %d %v %v %q", *workers, *status, *mdns, *boot)
	}
	os.WriteFile(p, []byte(`{"wokers":3}`), 0o600)
	c, _ = loadConfig(p)
	if err := c.apply(fs); err == nil {
		t.Fatal("unknown field accepted")
	}
}
