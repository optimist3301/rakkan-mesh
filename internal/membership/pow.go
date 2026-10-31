package membership

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/bits"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/libp2p/go-libp2p/core/peer"
)

// Proof is a guest's proof of work on its identity: a nonce such that
// SHA-256(domain, network ID, peer ID, nonce) has at least the policy's
// number of leading zero bits. It binds the peer to the network, so it
// cannot be reused elsewhere, and makes identities cost CPU time: a guest
// caught cheating loses the trust and credit tied to its identity, and a
// fresh identity costs this work again. Checking it is one hash.
type Proof struct {
	Nonce uint64 `json:"nonce"`
}

var powDomain = []byte("rakkan-pow-v1\x00")

func powPrefix(network string, p peer.ID) []byte {
	b := append([]byte{}, powDomain...)
	b = append(b, network...)
	b = append(b, 0)
	return append(b, p...)
}

func leadingZeros(sum [32]byte) int {
	n := 0
	for i := 0; i < len(sum); i += 8 {
		w := binary.BigEndian.Uint64(sum[i:])
		n += bits.LeadingZeros64(w)
		if w != 0 {
			break
		}
	}
	return n
}

// Work returns the leading zero bits of a proof's hash.
func (n *Network) Work(p peer.ID, pr Proof) int {
	b := binary.BigEndian.AppendUint64(powPrefix(n.ID, p), pr.Nonce)
	return leadingZeros(sha256.Sum256(b))
}

// Solve finds a proof of at least bitsWanted bits for p on all CPUs. The
// expected work is 2^bitsWanted hashes.
func (n *Network) Solve(ctx context.Context, p peer.ID, bitsWanted int) (Proof, error) {
	if bitsWanted < 0 || bitsWanted > MaxPowBits {
		return Proof{}, fmt.Errorf("proof of work of %d bits, limit %d", bitsWanted, MaxPowBits)
	}
	prefix := powPrefix(n.ID, p)
	workers := runtime.GOMAXPROCS(0)
	var found atomic.Bool
	var result atomic.Uint64
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			buf := binary.BigEndian.AppendUint64(append([]byte{}, prefix...), 0)
			at := len(buf) - 8
			for i, nonce := 0, uint64(w); ; i, nonce = i+1, nonce+uint64(workers) {
				if i%(1<<16) == 0 && (found.Load() || ctx.Err() != nil) {
					return
				}
				binary.BigEndian.PutUint64(buf[at:], nonce)
				if leadingZeros(sha256.Sum256(buf)) >= bitsWanted {
					if found.CompareAndSwap(false, true) {
						result.Store(nonce)
					}
					return
				}
			}
		})
	}
	wg.Wait()
	if !found.Load() {
		return Proof{}, ctx.Err()
	}
	return Proof{Nonce: result.Load()}, nil
}

// LoadProof reads a proof file.
func LoadProof(path string) (Proof, error) {
	var pr Proof
	b, err := os.ReadFile(path)
	if err != nil {
		return pr, err
	}
	err = json.Unmarshal(b, &pr)
	return pr, err
}

// SaveProof writes a proof file.
func SaveProof(path string, pr Proof) error {
	b, _ := json.Marshal(pr)
	return os.WriteFile(path, b, 0o644)
}
