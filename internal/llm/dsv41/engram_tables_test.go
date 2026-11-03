package dsv41

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/llm/bpe"
)

func readGzip(t testing.TB, path string) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(z)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The real tokenizer (testdata, gzipped), read once.
var realTokenizer = sync.OnceValues(func() (*bpe.Tokenizer, error) {
	f, err := os.Open("../testdata/dsv41-tokenizer.json.gz")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	b, err := io.ReadAll(z)
	if err != nil {
		return nil, err
	}
	return bpe.Read(b, 129280)
})

func tokenizer(t testing.TB) *bpe.Tokenizer {
	t.Helper()
	tok, err := realTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// realVectors are the reference's answers for the real model
// (tools/llmref/dsv41/vectors.py).
type realVectors struct {
	VocabSize          int         `json:"vocab_size"`
	Compressed         int         `json:"compressed_vocab_size"`
	Pad                int64       `json:"pad_id"`
	TokenMap           []int64     `json:"token_map"`
	Primes             [][][]int64 `json:"primes"`
	Offsets            [][]int64   `json:"offsets"`
	Multipliers        [][]int64   `json:"multipliers"`
	MultipliersByLayer struct {
		Bound  int64     `json:"bound"`
		Values [][]int64 `json:"values"`
	} `json:"multipliers_by_layer"`
	Hashes []struct {
		IDs     []int       `json:"ids"`
		HashIDs [][][]int64 `json:"hash_ids"` // [position][engram layer][column]
	} `json:"hashes"`
}

func loadRealVectors(t *testing.T) *realVectors {
	var v realVectors
	if err := json.Unmarshal(readGzip(t, "../testdata/dsv41-vectors.json.gz"), &v); err != nil {
		t.Fatal(err)
	}
	return &v
}

func TestNumpyIntegersMatchesNumpy(t *testing.T) {
	v := loadRealVectors(t)
	for layer, want := range v.MultipliersByLayer.Values {
		got, err := numpyIntegers(10007*uint64(layer), v.MultipliersByLayer.Bound, len(want))
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			got[i] = got[i]*2 + 1
		}
		if !slices.Equal(got, want) {
			t.Fatalf("seed %d: %v, numpy %v", 10007*layer, got, want)
		}
	}
}

// The real model's tables, built from its tokenizer and config, against
// the reference's: the compressed map of all 129280 tokens (Unicode
// normalization, case folding, whitespace), the primes, offsets and
// multipliers, and the hashes of the n-grams of sample texts.
func TestEngramTablesMatchReference(t *testing.T) {
	cfg, err := LoadConfig("../../../tools/llmref/dsv41/upstream/config.json")
	if err != nil {
		t.Fatal(err)
	}
	tab, err := BuildEngramTables(cfg, tokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	v := loadRealVectors(t)
	if tab.Compressed != v.Compressed || tab.Pad != v.Pad {
		t.Errorf("compressed %d pad %d, reference %d %d", tab.Compressed, tab.Pad, v.Compressed, v.Pad)
	}
	bad := 0
	for id, want := range v.TokenMap {
		if tab.TokenMap[id] != want {
			if bad++; bad <= 5 {
				t.Errorf("token %d (%q): compressed id %d, reference %d", id, tokenizer(t).Token(id), tab.TokenMap[id], want)
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d tokens map differently", bad, len(v.TokenMap))
	}
	for i, l := range tab.Layers {
		if !reflect.DeepEqual(l.Primes, v.Primes[i]) || !slices.Equal(l.Offsets, v.Offsets[i]) || !slices.Equal(l.Multipliers, v.Multipliers[i]) {
			t.Errorf("layer %d: primes, offsets or multipliers differ", l.Layer)
		}
	}
	for _, h := range v.Hashes {
		var hist []int64
		for pos, id := range h.IDs {
			hist = append(hist, tab.TokenMap[id])
			if len(hist) > tab.MaxNgram {
				hist = hist[1:]
			}
			if got := tab.hash(hist); !reflect.DeepEqual(got, h.HashIDs[pos]) {
				t.Fatalf("tokens %v position %d: hashes %v, reference %v", h.IDs, pos, got, h.HashIDs[pos])
			}
		}
	}
}

// The tiny fixture's tables, built the same way for its 512-token
// vocabulary, equal those the reference wrote for it.
func TestEngramTablesTinyMatchFixture(t *testing.T) {
	cfg, err := LoadConfig(fixture + "/config.json")
	if err != nil {
		t.Fatal(err)
	}
	got, err := BuildEngramTables(cfg, tokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	want, err := LoadEngramTables(fixture + "/engram.json")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tables differ from the fixture's engram.json")
	}
}
