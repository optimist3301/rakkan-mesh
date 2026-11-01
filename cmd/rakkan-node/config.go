package main

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/libp2p/go-libp2p/core/crypto"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

// A node's JSON config file holds everything a daemon needs in one place:
//
//	{
//	  "network": "12D3Koo...",
//	  "private_key": "<base64 of the node key>",
//	  "certificate": { "body": "...", "sig": "..." },
//	  "bootstrap": ["/ip4/203.0.113.7/udp/4001/quic-v1/p2p/12D3Koo..."],
//	  "workers": 4,
//	  "status": "1m"
//	}
//
// "private_key" and "certificate" are the node's key and its network
// certificate, embedded instead of read from the -key and -cert files. Every
// other field is named like a daemon flag and takes its place: strings,
// numbers and booleans as such, durations as strings ("30s"), lists of
// addresses as arrays. A flag given on the command line overrides the file.
// Because the file holds a private key it must be readable only by its owner.

// Reserved config fields that are not flags.
const (
	cfgPrivateKey  = "private_key"
	cfgCertificate = "certificate"
)

// nodeConfig is a parsed config file.
type nodeConfig struct {
	key    crypto.PrivKey
	cert   *membership.SignedCert
	fields map[string]json.RawMessage // flag settings
}

// loadConfig reads and validates a config file.
func loadConfig(path string) (*nodeConfig, error) {
	if runtime.GOOS != "windows" {
		st, err := os.Stat(path)
		if err != nil {
			return nil, err
		}
		if st.Mode().Perm()&0o077 != 0 {
			return nil, fmt.Errorf("config %s holds a private key but is accessible to others (mode %04o): chmod 600 %s", path, st.Mode().Perm(), path)
		}
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	var fields map[string]json.RawMessage
	if err := dec.Decode(&fields); err != nil {
		return nil, fmt.Errorf("config %s: %w", path, err)
	}
	if dec.More() {
		return nil, fmt.Errorf("config %s: trailing data after the JSON object", path)
	}
	c := &nodeConfig{fields: fields}
	if raw, ok := fields[cfgPrivateKey]; ok {
		delete(fields, cfgPrivateKey)
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, fmt.Errorf("config %s: %s must be a base64 string", path, cfgPrivateKey)
		}
		kb, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			return nil, fmt.Errorf("config %s: %s: %w", path, cfgPrivateKey, err)
		}
		if c.key, err = decodeConfigKey(kb); err != nil {
			return nil, fmt.Errorf("config %s: %s: %w", path, cfgPrivateKey, err)
		}
	}
	if raw, ok := fields[cfgCertificate]; ok {
		delete(fields, cfgCertificate)
		if string(bytes.TrimSpace(raw)) != "null" {
			if c.cert, err = membership.DecodeCert(raw); err != nil {
				return nil, fmt.Errorf("config %s: %s: %w", path, cfgCertificate, err)
			}
		}
	}
	return c, nil
}

func decodeConfigKey(b []byte) (crypto.PrivKey, error) {
	sk, err := crypto.UnmarshalPrivateKey(b)
	if err != nil {
		return nil, err
	}
	if sk.Type() != crypto.Ed25519 {
		return nil, errors.New("not Ed25519")
	}
	return sk, nil
}

// apply sets every flag the config names and the command line did not. It
// rejects fields that name no flag, so a typo cannot silently do nothing.
func (c *nodeConfig) apply(fs *flag.FlagSet) error {
	given := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { given[f.Name] = true })
	names := make([]string, 0, len(c.fields))
	for name := range c.fields {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		f := fs.Lookup(name)
		if f == nil || name == "config" || name == "key" || name == "cert" {
			return fmt.Errorf("config: unknown field %q", name)
		}
		if given[name] {
			continue
		}
		v, err := flagValue(c.fields[name])
		if err != nil {
			return fmt.Errorf("config: %s: %w", name, err)
		}
		if err := fs.Set(name, v); err != nil {
			return fmt.Errorf("config: %s: %w", name, err)
		}
	}
	return nil
}

// flagValue renders a JSON value the way it would be typed on a command line.
func flagValue(raw json.RawMessage) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return "", err
	}
	switch v := v.(type) {
	case string:
		return v, nil
	case json.Number:
		return v.String(), nil
	case bool:
		return strconv.FormatBool(v), nil
	case []any:
		parts := make([]string, len(v))
		for i, e := range v {
			s, ok := e.(string)
			if !ok {
				return "", errors.New("lists may only hold strings")
			}
			parts[i] = s
		}
		return strings.Join(parts, ","), nil
	}
	return "", errors.New("want a string, number, boolean or list of strings")
}

// configCmd implements "rakkan-node config": tools for config files.
func configCmd(args []string) int {
	if len(args) == 0 || args[0] != "init" {
		fmt.Fprint(os.Stderr, usage)
		return 2
	}
	flags := flag.NewFlagSet("config init", flag.ExitOnError)
	network := flags.String("network", "", "network ID (required)")
	keyPath := flags.String("key", "", "existing node key file to embed; a new key is generated when empty")
	certPath := flags.String("cert", "", "node certificate to embed; without one the config is for a guest")
	out := flags.String("out", "node.json", "config file to write; an existing file is not overwritten")
	flags.Parse(args[1:])

	n, err := membership.ParseNetwork(*network)
	if err != nil {
		return fail(fmt.Errorf("-network: %w", err))
	}
	var sk crypto.PrivKey
	if *keyPath != "" {
		sk, err = membership.LoadKey(*keyPath)
	} else {
		sk, _, err = crypto.GenerateEd25519Key(rand.Reader)
	}
	if err != nil {
		return fail(err)
	}
	kb, err := crypto.MarshalPrivateKey(sk)
	if err != nil {
		return fail(err)
	}
	doc := map[string]any{"network": n.ID, cfgPrivateKey: base64.StdEncoding.EncodeToString(kb)}
	if *certPath != "" {
		sc, err := membership.LoadCert(*certPath)
		if err != nil {
			return fail(err)
		}
		doc[cfgCertificate] = sc
	}
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return fail(err)
	}
	f, err := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fail(err)
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		f.Close()
		return fail(err)
	}
	if err := f.Close(); err != nil {
		return fail(err)
	}
	fmt.Fprintf(os.Stderr, "config written to %s (owner-only: it holds the node's private key); add other daemon settings to it as JSON fields named like the flags\n", *out)
	if *certPath == "" {
		fmt.Fprintln(os.Stderr, "no certificate: the node will join as a guest")
	}
	return 0
}
