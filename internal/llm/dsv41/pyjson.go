package dsv41

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// The chat format renders JSON values into prompts with Python's
// json.dumps, so a port must keep what Python keeps: the order of an
// object's keys (Python dicts preserve it), and numbers as Python reads
// and prints them. Value is a JSON value with those semantics.

// Kind is a JSON value's type.
type Kind int

const (
	Null Kind = iota
	Bool
	Number
	String
	Array
	Object
)

// Value is a JSON value. Objects keep their keys in order; numbers keep
// their literal (Python reads integers exactly and others as floats).
type Value struct {
	Kind Kind
	B    bool
	N    string // number literal
	S    string
	A    []*Value
	keys []string
	vals []*Value
}

func NewString(s string) *Value   { return &Value{Kind: String, S: s} }
func NewArray(a ...*Value) *Value { return &Value{Kind: Array, A: a} }
func NewObject() *Value           { return &Value{Kind: Object} }

// Keys lists an object's keys in order.
func (v *Value) Keys() []string { return v.keys }

// Get returns an object's member, or nil when absent.
func (v *Value) Get(key string) *Value {
	if v == nil || v.Kind != Object {
		return nil
	}
	for i, k := range v.keys {
		if k == key {
			return v.vals[i]
		}
	}
	return nil
}

// Has reports whether an object has a key (even if null).
func (v *Value) Has(key string) bool {
	return v != nil && v.Kind == Object && v.index(key) >= 0
}

func (v *Value) index(key string) int {
	for i, k := range v.keys {
		if k == key {
			return i
		}
	}
	return -1
}

// Set assigns a member as Python does: an existing key keeps its place, a
// new one goes last.
func (v *Value) Set(key string, x *Value) {
	if i := v.index(key); i >= 0 {
		v.vals[i] = x
		return
	}
	v.keys = append(v.keys, key)
	v.vals = append(v.vals, x)
}

// Delete removes a member, returning it (nil if absent).
func (v *Value) Delete(key string) *Value {
	i := v.index(key)
	if i < 0 {
		return nil
	}
	x := v.vals[i]
	v.keys = append(v.keys[:i:i], v.keys[i+1:]...)
	v.vals = append(v.vals[:i:i], v.vals[i+1:]...)
	return x
}

// Clone copies a value deeply (copy.deepcopy).
func (v *Value) Clone() *Value {
	if v == nil {
		return nil
	}
	c := *v
	c.A = nil
	for _, x := range v.A {
		c.A = append(c.A, x.Clone())
	}
	c.keys = append([]string(nil), v.keys...)
	c.vals = nil
	for _, x := range v.vals {
		c.vals = append(c.vals, x.Clone())
	}
	return &c
}

// ShallowClone copies an object's members but not their values (dict(d)).
func (v *Value) ShallowClone() *Value {
	c := *v
	c.keys = append([]string(nil), v.keys...)
	c.vals = append([]*Value(nil), v.vals...)
	c.A = append([]*Value(nil), v.A...)
	return &c
}

// isNone is Python's `x is None` for a member lookup: absent or null.
func (v *Value) isNone() bool { return v == nil || v.Kind == Null }

// Truthy is Python's truth value.
func (v *Value) Truthy() bool {
	if v == nil {
		return false
	}
	switch v.Kind {
	case Bool:
		return v.B
	case Number:
		f, _ := strconv.ParseFloat(v.N, 64)
		return f != 0
	case String:
		return v.S != ""
	case Array:
		return len(v.A) > 0
	case Object:
		return len(v.keys) > 0
	}
	return false
}

// isInt reports whether a number is a Python int.
func (v *Value) isInt() bool { return !strings.ContainsAny(v.N, ".eE") }

// ParseJSON parses JSON as Python's json.loads does: objects keep key
// order (a repeated key keeps its first place and its last value),
// surrounding whitespace is allowed, anything else after the value is not.
func ParseJSON(s string) (*Value, error) {
	p := &jsonParser{s: s}
	p.ws()
	v, err := p.value()
	if err != nil {
		return nil, err
	}
	p.ws()
	if p.i != len(s) {
		return nil, fmt.Errorf("json: extra data at %d", p.i)
	}
	return v, nil
}

type jsonParser struct {
	s string
	i int
}

func (p *jsonParser) ws() {
	for p.i < len(p.s) && strings.IndexByte(" \t\n\r", p.s[p.i]) >= 0 {
		p.i++
	}
}

var errJSON = errors.New("json: invalid")

func (p *jsonParser) value() (*Value, error) {
	if p.i >= len(p.s) {
		return nil, errJSON
	}
	switch c := p.s[p.i]; {
	case c == '{':
		p.i++
		v := NewObject()
		p.ws()
		if p.i < len(p.s) && p.s[p.i] == '}' {
			p.i++
			return v, nil
		}
		for {
			p.ws()
			if p.i >= len(p.s) || p.s[p.i] != '"' {
				return nil, errJSON
			}
			k, err := p.str()
			if err != nil {
				return nil, err
			}
			p.ws()
			if p.i >= len(p.s) || p.s[p.i] != ':' {
				return nil, errJSON
			}
			p.i++
			p.ws()
			x, err := p.value()
			if err != nil {
				return nil, err
			}
			v.Set(k, x)
			p.ws()
			if p.i < len(p.s) && p.s[p.i] == ',' {
				p.i++
				continue
			}
			if p.i < len(p.s) && p.s[p.i] == '}' {
				p.i++
				return v, nil
			}
			return nil, errJSON
		}
	case c == '[':
		p.i++
		v := NewArray()
		p.ws()
		if p.i < len(p.s) && p.s[p.i] == ']' {
			p.i++
			return v, nil
		}
		for {
			p.ws()
			x, err := p.value()
			if err != nil {
				return nil, err
			}
			v.A = append(v.A, x)
			p.ws()
			if p.i < len(p.s) && p.s[p.i] == ',' {
				p.i++
				continue
			}
			if p.i < len(p.s) && p.s[p.i] == ']' {
				p.i++
				return v, nil
			}
			return nil, errJSON
		}
	case c == '"':
		s, err := p.str()
		return NewString(s), err
	case strings.HasPrefix(p.s[p.i:], "true"):
		p.i += 4
		return &Value{Kind: Bool, B: true}, nil
	case strings.HasPrefix(p.s[p.i:], "false"):
		p.i += 5
		return &Value{Kind: Bool}, nil
	case strings.HasPrefix(p.s[p.i:], "null"):
		p.i += 4
		return &Value{Kind: Null}, nil
	case c == '-' || c >= '0' && c <= '9':
		return p.number()
	}
	return nil, errJSON
}

func (p *jsonParser) number() (*Value, error) {
	start := p.i
	if p.s[p.i] == '-' {
		p.i++
	}
	digits := func() int {
		n := 0
		for p.i < len(p.s) && p.s[p.i] >= '0' && p.s[p.i] <= '9' {
			p.i++
			n++
		}
		return n
	}
	if p.i < len(p.s) && p.s[p.i] == '0' {
		p.i++
	} else if digits() == 0 {
		return nil, errJSON
	}
	if p.i < len(p.s) && p.s[p.i] == '.' {
		p.i++
		if digits() == 0 {
			return nil, errJSON
		}
	}
	if p.i < len(p.s) && (p.s[p.i] == 'e' || p.s[p.i] == 'E') {
		p.i++
		if p.i < len(p.s) && (p.s[p.i] == '+' || p.s[p.i] == '-') {
			p.i++
		}
		if digits() == 0 {
			return nil, errJSON
		}
	}
	return &Value{Kind: Number, N: p.s[start:p.i]}, nil
}

func (p *jsonParser) str() (string, error) {
	p.i++ // opening quote
	var b strings.Builder
	for p.i < len(p.s) {
		c := p.s[p.i]
		switch {
		case c == '"':
			p.i++
			return b.String(), nil
		case c < 0x20:
			return "", errJSON // json.loads is strict about control characters
		case c == '\\':
			if p.i+1 >= len(p.s) {
				return "", errJSON
			}
			e := p.s[p.i+1]
			p.i += 2
			switch e {
			case '"', '\\', '/':
				b.WriteByte(e)
			case 'b':
				b.WriteByte('\b')
			case 'f':
				b.WriteByte('\f')
			case 'n':
				b.WriteByte('\n')
			case 'r':
				b.WriteByte('\r')
			case 't':
				b.WriteByte('\t')
			case 'u':
				r, ok := p.hex4()
				if !ok {
					return "", errJSON
				}
				if utf16.IsSurrogate(r) {
					if strings.HasPrefix(p.s[p.i:], "\\u") {
						save := p.i
						p.i += 2
						if r2, ok := p.hex4(); ok {
							if d := utf16.DecodeRune(r, r2); d != utf8.RuneError {
								b.WriteRune(d)
								continue
							}
						}
						p.i = save
					}
					r = utf8.RuneError // a lone surrogate: Go strings cannot hold it
				}
				b.WriteRune(r)
			default:
				return "", errJSON
			}
		default:
			b.WriteByte(c)
			p.i++
		}
	}
	return "", errJSON
}

func (p *jsonParser) hex4() (rune, bool) {
	if p.i+4 > len(p.s) {
		return 0, false
	}
	n, err := strconv.ParseUint(p.s[p.i:p.i+4], 16, 32)
	if err != nil {
		return 0, false
	}
	p.i += 4
	return rune(n), true
}

// Dumps is Python's json.dumps(v, ensure_ascii=False): ", " and ": "
// separators, non-ASCII characters as they are.
func (v *Value) Dumps() string {
	var b strings.Builder
	v.dump(&b)
	return b.String()
}

func (v *Value) dump(b *strings.Builder) {
	if v == nil {
		b.WriteString("null")
		return
	}
	switch v.Kind {
	case Null:
		b.WriteString("null")
	case Bool:
		if v.B {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case Number:
		b.WriteString(v.pyNumber(true))
	case String:
		dumpString(b, v.S)
	case Array:
		b.WriteByte('[')
		for i, x := range v.A {
			if i > 0 {
				b.WriteString(", ")
			}
			x.dump(b)
		}
		b.WriteByte(']')
	case Object:
		b.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				b.WriteString(", ")
			}
			dumpString(b, k)
			b.WriteString(": ")
			v.vals[i].dump(b)
		}
		b.WriteByte('}')
	}
}

func dumpString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pyNumber prints a number as Python prints what json.loads made of it:
// integers exactly, floats by repr (json: Infinity and NaN for overflow).
func (v *Value) pyNumber(json bool) string {
	if v.isInt() {
		n, ok := new(big.Int).SetString(v.N, 10)
		if !ok {
			return v.N
		}
		return n.String()
	}
	f, err := strconv.ParseFloat(v.N, 64)
	if err != nil && !math.IsInf(f, 0) {
		return v.N
	}
	switch {
	case math.IsInf(f, 1):
		if json {
			return "Infinity"
		}
		return "inf"
	case math.IsInf(f, -1):
		if json {
			return "-Infinity"
		}
		return "-inf"
	}
	return pyFloatRepr(f)
}

// pyFloatRepr is Python's repr of a float: the shortest digits that round
// back, in positional notation for decimal exponents from -4 to 15 (with
// ".0" when whole), else in exponent notation with at least two exponent
// digits.
func pyFloatRepr(f float64) string {
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	s := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	sign := ""
	if s[0] == '-' {
		sign, s = "-", s[1:]
	}
	mant, expStr, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(expStr)
	digits := strings.Replace(mant, ".", "", 1)
	decpt := exp + 1 // digits are 0.d1d2... × 10^decpt
	if decpt > -4 && decpt <= 16 {
		switch {
		case decpt <= 0:
			return sign + "0." + strings.Repeat("0", -decpt) + digits
		case decpt >= len(digits):
			return sign + digits + strings.Repeat("0", decpt-len(digits)) + ".0"
		default:
			return sign + digits[:decpt] + "." + digits[decpt:]
		}
	}
	m := digits[:1]
	if len(digits) > 1 {
		m += "." + digits[1:]
	}
	e := decpt - 1
	es := "+"
	if e < 0 {
		es, e = "-", -e
	}
	return fmt.Sprintf("%s%se%s%02d", sign, m, es, e)
}

// pyStr is Python's str() of a scalar, as str.format renders it; lists and
// dicts would render as Python reprs, which prompts never contain.
func pyStr(v *Value) (string, error) {
	switch {
	case v.isNone():
		return "None", nil
	case v.Kind == String:
		return v.S, nil
	case v.Kind == Bool:
		if v.B {
			return "True", nil
		}
		return "False", nil
	case v.Kind == Number:
		return v.pyNumber(false), nil
	}
	return "", errors.New("chat: a list or object where text belongs")
}
