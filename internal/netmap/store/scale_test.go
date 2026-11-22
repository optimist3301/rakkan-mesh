package store

import (
	"bytes"
	"crypto/ed25519"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

func BenchmarkLoad(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "m.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	k := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{1}, 32))
	var list []*obs.Observation
	for i := range 100000 {
		o, _ := obs.Sign(obs.Body{Time: int64(i + 1), Mode: obs.ModeActive, Method: "neigh",
			Subject: []string{fmt.Sprintf("mac:00:11:32:%02x:%02x:%02x", i>>16, (i>>8)&0xff, i&0xff), fmt.Sprintf("ip:nx/10.0.%d.%d", i>>8&0xff, i&0xff)}}, k)
		list = append(list, o)
	}
	st.Add(list)
	b.ResetTimer()
	for b.Loop() {
		if _, err := st.All(); err != nil {
			b.Fatal(err)
		}
	}
}
