package mldht

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strconv"
)

// Bencode (BEP 3): integers, byte strings, lists and dictionaries with
// sorted keys. Values decode to int64, []byte, []any and map[string]any.

// maxDepth bounds nesting when decoding what strangers send.
const maxDepth = 16

var errBencode = errors.New("mldht: malformed bencode")

// encode appends v's bencoding to b. It takes the decoded types, plus int,
// string and [][]byte.
func encode(b []byte, v any) []byte {
	switch x := v.(type) {
	case int:
		return encode(b, int64(x))
	case int64:
		b = append(b, 'i')
		b = strconv.AppendInt(b, x, 10)
		return append(b, 'e')
	case string:
		return encode(b, []byte(x))
	case []byte:
		b = strconv.AppendInt(b, int64(len(x)), 10)
		b = append(b, ':')
		return append(b, x...)
	case [][]byte:
		b = append(b, 'l')
		for _, e := range x {
			b = encode(b, e)
		}
		return append(b, 'e')
	case []any:
		b = append(b, 'l')
		for _, e := range x {
			b = encode(b, e)
		}
		return append(b, 'e')
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b = append(b, 'd')
		for _, k := range keys {
			b = encode(b, k)
			b = encode(b, x[k])
		}
		return append(b, 'e')
	}
	panic(fmt.Sprintf("mldht: cannot bencode %T", v))
}

// decode reads one value from the start of b and returns the rest.
func decode(b []byte) (any, []byte, error) { return decodeDepth(b, 0) }

func decodeDepth(b []byte, depth int) (any, []byte, error) {
	if len(b) == 0 || depth > maxDepth {
		return nil, nil, errBencode
	}
	switch c := b[0]; {
	case c == 'i':
		end := bytes.IndexByte(b, 'e')
		if end < 2 {
			return nil, nil, errBencode
		}
		s := string(b[1:end])
		if s == "-0" || len(s) > 1 && (s[0] == '0' || s[0] == '-' && s[1] == '0') {
			return nil, nil, errBencode
		}
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, nil, errBencode
		}
		return n, b[end+1:], nil
	case c >= '0' && c <= '9':
		colon := bytes.IndexByte(b, ':')
		if colon < 1 || colon > 7 || b[0] == '0' && colon > 1 {
			return nil, nil, errBencode
		}
		n, err := strconv.Atoi(string(b[:colon]))
		if err != nil || n > len(b)-colon-1 {
			return nil, nil, errBencode
		}
		return b[colon+1 : colon+1+n], b[colon+1+n:], nil
	case c == 'l':
		var list []any
		b = b[1:]
		for {
			if len(b) == 0 {
				return nil, nil, errBencode
			}
			if b[0] == 'e' {
				return list, b[1:], nil
			}
			v, rest, err := decodeDepth(b, depth+1)
			if err != nil {
				return nil, nil, err
			}
			list = append(list, v)
			b = rest
		}
	case c == 'd':
		dict := map[string]any{}
		b = b[1:]
		for {
			if len(b) == 0 {
				return nil, nil, errBencode
			}
			if b[0] == 'e' {
				return dict, b[1:], nil
			}
			k, rest, err := decodeDepth(b, depth+1)
			key, ok := k.([]byte)
			if err != nil || !ok {
				return nil, nil, errBencode
			}
			if _, dup := dict[string(key)]; dup {
				return nil, nil, errBencode
			}
			v, rest, err := decodeDepth(rest, depth+1)
			if err != nil {
				return nil, nil, err
			}
			dict[string(key)] = v
			b = rest
		}
	}
	return nil, nil, errBencode
}

// decodeAll decodes b, which must hold exactly one value.
func decodeAll(b []byte) (any, error) {
	v, rest, err := decode(b)
	if err == nil && len(rest) > 0 {
		err = errBencode
	}
	return v, err
}

// Accessors of decoded dictionaries: zero values when absent or of
// another type.
func dictOf(v any) map[string]any { d, _ := v.(map[string]any); return d }
func bytesAt(d map[string]any, k string) []byte {
	b, _ := d[k].([]byte)
	return b
}
func intAt(d map[string]any, k string) (int64, bool) {
	n, ok := d[k].(int64)
	return n, ok
}
