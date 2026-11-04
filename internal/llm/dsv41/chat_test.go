package dsv41

import (
	"encoding/json"
	"testing"
)

// The chat format against the reference's encoding.py
// (tools/llmref/dsv41/chat_vectors.py): every case's prompt and images, or
// its failure, and every completion's parse.
func TestChatMatchesReference(t *testing.T) {
	root, err := ParseJSON(string(readGzip(t, "../testdata/dsv41-chat.json.gz")))
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range root.Get("encode").A {
		name := c.Get("name").S
		o := ChatOptions{ThinkingMode: c.Get("thinking_mode").S, ReasoningEffort: c.Get("reasoning_effort")}
		kw := c.Get("kwargs")
		if ctx := kw.Get("context"); ctx != nil {
			o.Context = ctx.A
		}
		if d := kw.Get("drop_thinking"); d != nil && !d.B {
			o.KeepThinking = true
		}
		if b := kw.Get("add_default_bos_token"); b != nil && !b.B {
			o.NoBOS = true
		}
		prompt, images, err := EncodeChat(c.Get("messages").A, o)
		label := func() string {
			return name + "/" + o.ThinkingMode + "/" + o.ReasoningEffort.Dumps()
		}
		if want := c.Get("error"); want != nil {
			if err == nil {
				t.Errorf("case %d %s: no error, reference raises %s", i, label(), want.S)
			}
			continue
		}
		if err != nil {
			t.Errorf("case %d %s: %v", i, label(), err)
			continue
		}
		if want := c.Get("prompt").S; prompt != want {
			t.Errorf("case %d %s: prompt\n%q\nreference\n%q", i, label(), prompt, want)
			continue
		}
		if got, want := NewArray(images...).Dumps(), c.Get("images").Dumps(); got != want {
			t.Errorf("case %d %s: images %s, reference %s", i, label(), got, want)
		}
	}
	for i, c := range root.Get("parse").A {
		msg, err := ParseCompletion(c.Get("text").S, c.Get("thinking_mode").S)
		if want := c.Get("error"); want != nil {
			if err == nil {
				t.Errorf("parse %d: no error, reference raises %s", i, want.S)
			}
			continue
		}
		if err != nil {
			t.Errorf("parse %d: %v", i, err)
			continue
		}
		want := c.Get("message")
		if msg.Content != want.Get("content").S || msg.ReasoningContent != want.Get("reasoning_content").S ||
			NewArray(msg.ToolCalls...).Dumps() != want.Get("tool_calls").Dumps() {
			t.Errorf("parse %d: %+v %s, reference %s", i, msg, NewArray(msg.ToolCalls...).Dumps(), want.Dumps())
		}
	}
}

func TestPyFloatRepr(t *testing.T) {
	for in, want := range map[string]string{
		"1.0": "1.0", "1.5": "1.5", "1e16": "1e+16", "1e15": "1000000000000000.0", "0.0001": "0.0001",
		"1e-05": "1e-05", "2.5e-300": "2.5e-300", "-0.0": "-0.0", "0.3333333333333333": "0.3333333333333333",
		"1.7976931348623157e308": "1.7976931348623157e+308", "1e22": "1e+22", "123.456": "123.456",
		"100": "100", "-0": "0", "1E2": "100.0", "123456789012345678901234567890": "123456789012345678901234567890",
	} {
		v, err := ParseJSON(in)
		if err != nil {
			t.Fatal(err)
		}
		if got := v.Dumps(); got != want {
			t.Errorf("%s dumps as %s, want %s", in, got, want)
		}
	}
	_ = json.Valid
}
