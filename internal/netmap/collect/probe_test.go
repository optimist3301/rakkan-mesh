//go:build linux

package collect

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"net"
	"testing"

	"golang.org/x/crypto/ssh"
)

// The probe must capture the server's host key without authenticating.
func TestSSHHostKey(t *testing.T) {
	debugSSH = func(err error) { t.Log("handshake:", err) }
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	signer, err := ssh.NewSignerFromKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &ssh.ServerConfig{ServerVersion: "SSH-2.0-TestServer_1.0",
		PasswordCallback: func(ssh.ConnMetadata, []byte) (*ssh.Permissions, error) { return nil, errors.New("no") }}
	cfg.AddHostKey(signer)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err == nil {
			ssh.NewServerConn(c, cfg)
			c.Close()
		}
	}()
	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	fp, banner := sshHostKey(conn, ln.Addr().String())
	if want := ssh.FingerprintSHA256(signer.PublicKey()); fp != want {
		t.Fatalf("fingerprint %q, want %q", fp, want)
	}
	if banner != "SSH-2.0-TestServer_1.0" {
		t.Fatalf("banner %q", banner)
	}
}
