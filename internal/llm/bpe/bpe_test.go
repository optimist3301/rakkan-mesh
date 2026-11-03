package bpe

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"slices"
	"strings"
	"testing"
)

func readGzip(t *testing.T, path string) []byte {
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

// DeepSeek-V4.1's tokenizer against the tokenizers library's answers
// (tools/llmref/dsv41/vectors.py): chosen texts, and random strings mixing
// every character class its three splitting patterns tell apart.
func TestDeepSeekMatchesReference(t *testing.T) {
	tok, err := Read(readGzip(t, "../testdata/dsv41-tokenizer.json.gz"), 129280)
	if err != nil {
		t.Fatal(err)
	}
	type tc struct {
		Text string
		IDs  []int `json:"ids"`
	}
	var v struct {
		Tokenize       []tc `json:"tokenize"`
		TokenizeRandom []tc `json:"tokenize_random"`
	}
	if err := json.Unmarshal(readGzip(t, "../testdata/dsv41-vectors.json.gz"), &v); err != nil {
		t.Fatal(err)
	}
	cases := append(v.Tokenize, v.TokenizeRandom...)
	if len(cases) < 400 {
		t.Fatalf("%d cases", len(cases))
	}
	bad := 0
	for _, c := range cases {
		got := tok.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			if bad++; bad <= 5 {
				t.Errorf("%q:\n got %v\nwant %v", c.Text, got, c.IDs)
			}
			continue
		}
		var b strings.Builder
		for _, id := range got {
			b.Write(tok.Piece(0, id))
		}
		if b.String() != c.Text {
			t.Errorf("%q decodes to %q", c.Text, b.String())
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d texts tokenize differently", bad, len(cases))
	}
}
