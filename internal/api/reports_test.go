package api

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/sealed"
)

func TestReportsAPI(t *testing.T) {
	root, n, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	sk, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	id, _ := peer.IDFromPrivateKey(sk)
	cert, _ := membership.Issue(root, nil, membership.Cert{Network: n.ID, Subject: id, Roles: []string{membership.RoleMember},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour)})
	cfg := mesh.Config{Key: sk, Network: n, Cert: cert, ListenAddrs: []string{"/ip4/127.0.0.1/udp/0/quic-v1"}}

	// Without a spool the node refuses.
	off, err := mesh.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer off.Close()
	srv := httptest.NewServer((&Server{Node: off, Network: n.ID}).Handler())
	defer srv.Close()
	res, _ := http.Post(srv.URL+"/v1/reports", "application/octet-stream", bytes.NewReader([]byte("x")))
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("reports off: %d", res.StatusCode)
	}
	res.Body.Close()

	cfg.Reports.Dir = t.TempDir()
	node, err := mesh.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer node.Close()
	srv2 := httptest.NewServer((&Server{Node: node, Network: n.ID}).Handler())
	defer srv2.Close()

	text := []byte(`{"devices":["printer"]}`)
	res, err = http.Post(srv2.URL+"/v1/reports", "application/octet-stream", bytes.NewReader(text))
	if err != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("post: %v %v", res, err)
	}
	res.Body.Close()

	var got struct {
		Network string
		Reports []struct {
			Seq    int64
			ID     string
			Packet []byte
		}
	}
	for _, after := range []string{"", "0"} {
		res, err = http.Get(srv2.URL + "/v1/reports?after=" + after)
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(res.Body)
		res.Body.Close()
		if err := json.Unmarshal(body, &got); err != nil || len(got.Reports) != 1 || got.Network != n.ID {
			t.Fatalf("reports: %s %v", body, err)
		}
	}
	if bytes.Contains(got.Reports[0].Packet, []byte("printer")) {
		t.Fatal("the API hands out plaintext")
	}
	pkt, _, err := mesh.VerifyReport(n, got.Reports[0].Packet, time.Now(), nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := root.Raw()
	plain, err := sealed.Open(ed25519.PrivateKey(raw), pkt.Sealed, mesh.ReportContext(n.ID))
	if err != nil || !bytes.Equal(plain, text) {
		t.Fatalf("root opens: %q %v", plain, err)
	}
	res, _ = http.Get(srv2.URL + "/v1/reports?after=" + string(rune('0'+9)) + "99999999999999999999")
	if res.StatusCode != http.StatusBadRequest {
		// an overflowing cursor is the caller's mistake, not an empty page
		t.Fatalf("bad cursor: %d", res.StatusCode)
	}
	res.Body.Close()
}
