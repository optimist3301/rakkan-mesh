package dsv41

import (
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// DeepSeek-V4.1's chat format: messages to prompt text, and a completion
// back to a message. A port of the reference's encoding/encoding.py,
// operating on JSON values with Python's semantics (pyjson.go) so that the
// same messages give the same prompt byte for byte.

const (
	bosToken           = "<｜begin▁of▁sentence｜>"
	eosToken           = "<｜end▁of▁sentence｜>"
	thinkingStartToken = "<think>"
	thinkingEndToken   = "</think>"
	dsmlToken          = "｜DSML｜"
	userSP             = "<｜User｜>"
	assistantSP        = "<｜Assistant｜>"
	latestReminderSP   = "<｜latest_reminder｜>"
	systemSP           = "<｜System｜>"
	imagePlaceholder   = "<｜deepseek_image｜>"

	toolCallsBlockName   = " calls"
	toolCallTagName      = " invoke"
	toolParameterTagName = " parameter"
)

var taskTokens = map[string]string{
	"action":    "<｜action｜>",
	"query":     "<｜query｜>",
	"authority": "<｜authority｜>",
	"domain":    "<｜domain｜>",
	"title":     "<｜title｜>",
	"read_url":  "<｜read_url｜>",
}

var reasoningEfforts = map[string]int{"low": 50, "high": 75, "max": 100}

const toolsTemplate = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml_token}{tc_block_name}>\" block like the following:\n\n<{dsml_token}{tc_block_name}>\n<{dsml_token}{tool_call_tag_name} name=\"$TOOL_NAME\">\n<{dsml_token}{tool_parameter_tag_name} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>\n...\n</{dsml_token}{tool_call_tag_name}>\n<{dsml_token}{tool_call_tag_name} name=\"$TOOL_NAME2\">\n...\n</{dsml_token}{tool_call_tag_name}>\n</{dsml_token}{tc_block_name}>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.\n\nOtherwise, output directly after {thinking_end_token} with tool calls or final response.\n\n### Available Tool Schemas\n\n{tool_schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"

// ChatOptions are encode_messages' keyword arguments.
type ChatOptions struct {
	// ThinkingMode is "chat" or "thinking".
	ThinkingMode string
	// ReasoningEffort is 1-100, "low", "high" or "max"; nil means "high".
	ReasoningEffort *Value
	// Context are earlier messages whose prompt is already cached: their
	// text is not rendered again.
	Context []*Value
	// KeepThinking keeps earlier turns' reasoning (drop_thinking=False).
	KeepThinking bool
	// NoBOS leaves out the begin-of-sentence token (add_default_bos_token=False).
	NoBOS bool
}

// ImageRecord is an image a prompt refers to, in order: its "url",
// "source" or "data" as the message gave it.
type ImageRecord = *Value

// EncodeChat renders messages (OpenAI-format JSON objects) into a prompt,
// and lists the images whose placeholders it contains.
func EncodeChat(messages []*Value, o ChatOptions) (string, []ImageRecord, error) {
	if o.ThinkingMode != "chat" && o.ThinkingMode != "thinking" {
		return "", nil, fmt.Errorf("chat: thinking mode %q", o.ThinkingMode)
	}
	for _, m := range append(slices.Clone(o.Context), messages...) {
		if err := validateNoImageTokens(m); err != nil {
			return "", nil, err
		}
	}
	context := mergeToolMessages(o.Context)
	msgs := mergeToolMessages(messages)
	// Sorting works on the messages in place, context included.
	sortToolResults(append(slices.Clone(context), msgs...))
	var err error
	var procContext []*Value
	if len(context) > 0 {
		if procContext, _, err = processImageMessages(context); err != nil {
			return "", nil, err
		}
	}
	procMsgs, images, err := processImageMessages(msgs)
	if err != nil {
		return "", nil, err
	}
	prompt, err := encodeMessagesText(procMsgs, o, procContext)
	return prompt, images, err
}

func isString(v *Value) bool { return v != nil && v.Kind == String }

func validateNoImageTokens(m *Value) error {
	if c := m.Get("content"); isString(c) && strings.Contains(c.S, imagePlaceholder) {
		return errors.New("chat: message content contains the image special token; images go in image blocks")
	}
	if r := m.Get("reasoning_content"); isString(r) && strings.Contains(r.S, imagePlaceholder) {
		return errors.New("chat: reasoning_content contains the image special token")
	}
	return nil
}

func textBlock(s *Value) *Value {
	b := NewObject()
	b.Set("type", NewString("text"))
	b.Set("text", s)
	return b
}

// mergeToolMessages turns "tool" messages into tool_result blocks of user
// messages, and merges consecutive user messages.
func mergeToolMessages(messages []*Value) []*Value {
	var merged []*Value
	lastUserWithBlocks := func() *Value {
		if len(merged) > 0 {
			l := merged[len(merged)-1]
			if r := l.Get("role"); isString(r) && r.S == "user" && l.Has("content_blocks") {
				return l
			}
		}
		return nil
	}
	for _, m := range messages {
		m = m.Clone()
		role := m.Get("role")
		switch {
		case isString(role) && role.S == "tool":
			content := m.Get("content")
			if !m.Has("content") {
				content = NewString("")
			}
			if m.Get("content_blocks").Truthy() && !isString(m.Get("content")) {
				content = m.Get("content_blocks")
			}
			id := m.Get("tool_call_id")
			if !m.Has("tool_call_id") {
				id = NewString("")
			}
			block := NewObject()
			block.Set("type", NewString("tool_result"))
			block.Set("tool_use_id", id)
			block.Set("content", content)
			if l := lastUserWithBlocks(); l != nil {
				l.Get("content_blocks").A = append(l.Get("content_blocks").A, block)
			} else {
				u := NewObject()
				u.Set("role", NewString("user"))
				u.Set("content_blocks", NewArray(block))
				merged = append(merged, u)
			}
		case isString(role) && role.S == "user":
			blocks := m.Get("content_blocks")
			if blocks.isNone() {
				content := m.Get("content")
				if !m.Has("content") {
					content = NewString("")
				}
				if content != nil && content.Kind == Array {
					blocks = content
				} else {
					blocks = NewArray(textBlock(content))
				}
			}
			if l := lastUserWithBlocks(); l != nil && l.Get("task").isNone() {
				l.Get("content_blocks").A = append(l.Get("content_blocks").A, blocks.A...)
			} else {
				m.Set("content_blocks", blocks)
				merged = append(merged, m)
			}
		default:
			merged = append(merged, m)
		}
	}
	return merged
}

// sortToolResults orders each user message's tool_result blocks by the
// order of the calls in the latest assistant message with calls.
func sortToolResults(messages []*Value) {
	order := map[string]int{}
	for _, m := range messages {
		role := m.Get("role")
		switch {
		case isString(role) && role.S == "assistant" && m.Get("tool_calls").Truthy():
			order = map[string]int{}
			for i, tc := range m.Get("tool_calls").A {
				id := tc.Get("id")
				if !id.Truthy() {
					id = tc.Get("function").Get("id")
				}
				if id.Truthy() && isString(id) {
					order[id.S] = i
				}
			}
		case isString(role) && role.S == "user" && m.Get("content_blocks").Truthy():
			blocks := m.Get("content_blocks").A
			var tools []*Value
			for _, b := range blocks {
				if t := b.Get("type"); isString(t) && t.S == "tool_result" {
					tools = append(tools, b)
				}
			}
			if len(tools) > 1 && len(order) > 0 {
				key := func(b *Value) int {
					if id := b.Get("tool_use_id"); isString(id) {
						return order[id.S]
					}
					return 0
				}
				slices.SortStableFunc(tools, func(a, b *Value) int { return key(a) - key(b) })
				k := 0
				var out []*Value
				for _, b := range blocks {
					if t := b.Get("type"); isString(t) && t.S == "tool_result" {
						out = append(out, tools[k])
						k++
					} else {
						out = append(out, b)
					}
				}
				m.Set("content_blocks", NewArray(out...))
			}
		}
	}
}

func isImageBlock(b *Value) bool {
	t := b.Get("type")
	return b.Kind == Object && isString(t) && (t.S == "image" || t.S == "image_url")
}

func extractImage(b *Value) (*Value, error) {
	rec := NewObject()
	rec.Set("type", NewString("image"))
	if b.Get("type").S == "image_url" {
		u := b.Get("image_url")
		if isString(u) {
			rec.Set("url", u)
		} else if v := u.Get("url"); v != nil {
			rec.Set("url", v)
		} else {
			rec.Set("url", NewString(""))
		}
	} else {
		for _, k := range []string{"source", "url", "data"} {
			if b.Has(k) {
				rec.Set(k, b.Get(k))
			}
		}
	}
	if !rec.Get("source").Truthy() && !rec.Get("url").Truthy() && !rec.Get("data").Truthy() {
		return nil, errors.New("chat: image block without a source")
	}
	return rec, nil
}

func processImageBlocks(blocks []*Value) ([]*Value, []*Value, error) {
	var out, images []*Value
	for _, b := range blocks {
		if b.Kind != Object {
			out = append(out, b)
			continue
		}
		t := b.Get("type")
		switch {
		case isImageBlock(b):
			out = append(out, textBlock(NewString(imagePlaceholder)))
			rec, err := extractImage(b)
			if err != nil {
				return nil, nil, err
			}
			images = append(images, rec)
		case isString(t) && t.S == "tool_result" && b.Get("content") != nil && b.Get("content").Kind == Array:
			b = b.ShallowClone()
			inner, imgs, err := processImageBlocks(b.Get("content").A)
			if err != nil {
				return nil, nil, err
			}
			b.Set("content", NewArray(inner...))
			out = append(out, b)
			images = append(images, imgs...)
		case isString(t) && t.S == "text":
			if text := b.Get("text"); isString(text) && strings.Contains(text.S, imagePlaceholder) {
				return nil, nil, errors.New("chat: text block contains the image placeholder")
			}
			out = append(out, b)
		default:
			out = append(out, b)
		}
	}
	return out, images, nil
}

// processImageMessages replaces image blocks by placeholders and collects
// their records in prompt order.
func processImageMessages(messages []*Value) ([]*Value, []*Value, error) {
	var out, images []*Value
	for _, m := range messages {
		m = m.Clone()
		if err := validateNoImageTokens(m); err != nil {
			return nil, nil, err
		}
		if c := m.Get("content"); c != nil && c.Kind == Array && !m.Has("content_blocks") {
			m.Set("content_blocks", m.Delete("content"))
		}
		if m.Get("content_blocks").Truthy() {
			blocks, imgs, err := processImageBlocks(m.Get("content_blocks").A)
			if err != nil {
				return nil, nil, err
			}
			m.Set("content_blocks", NewArray(blocks...))
			images = append(images, imgs...)
			if !isString(m.Get("content")) {
				var texts []string
				for _, b := range blocks {
					if t := b.Get("type"); b.Kind == Object && isString(t) && t.S == "text" {
						s, err := blockText(b)
						if err != nil {
							return nil, nil, err
						}
						texts = append(texts, s)
					}
				}
				m.Set("content", NewString(strings.Join(texts, "\n\n")))
			}
		}
		out = append(out, m)
	}
	return out, images, nil
}

// blockText is block.get("text", "") where a string is needed.
func blockText(b *Value) (string, error) {
	t := b.Get("text")
	if !b.Has("text") {
		return "", nil
	}
	if !isString(t) {
		return "", errors.New("chat: a text block's text is not a string")
	}
	return t.S, nil
}

func roleIs(m *Value, role string) bool { r := m.Get("role"); return isString(r) && r.S == role }

// lastUserIndex is the index of the last message starting a turn: a user
// message, or a system message after the first, unless it holds only tool
// results.
func lastUserIndex(messages []*Value) int {
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if roleIs(m, "user") || roleIs(m, "system") && i > 0 {
			if blocks := m.Get("content_blocks"); blocks.Truthy() {
				only := true
				for _, b := range blocks.A {
					if t := b.Get("type"); isString(t) && (t.S == "text" || t.S == "image" || t.S == "image_url") {
						only = false
					}
				}
				if only {
					continue
				}
			}
			return i
		}
	}
	return -1
}

func renderReasoningEffort(index int, mode string, effort *Value) (string, error) {
	budget := 75
	switch {
	case effort.isNone():
	case effort.Kind == Number && effort.isInt():
		var n int
		if _, err := fmt.Sscan(effort.N, &n); err != nil || n < 1 || n > 100 {
			return "", fmt.Errorf("chat: reasoning effort %s, want 1-100 or low, high, max", effort.N)
		}
		budget = n
	case isString(effort):
		b, ok := reasoningEfforts[effort.S]
		if !ok {
			return "", fmt.Errorf("chat: reasoning effort %q", effort.S)
		}
		budget = b
	default:
		return "", errors.New("chat: reasoning effort must be a number or a name")
	}
	if index == 0 && mode == "thinking" {
		return fmt.Sprintf("Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n", budget), nil
	}
	return "", nil
}

// splitToolName splits "namespace::name", checking it against an explicit
// namespace.
func splitToolName(name string, namespace *Value) (*Value, string, error) {
	if prefix, bare, ok := strings.Cut(name, "::"); ok {
		if !namespace.isNone() && !(isString(namespace) && namespace.S == prefix) {
			return nil, "", fmt.Errorf("chat: conflicting tool namespaces in %q", name)
		}
		namespace, name = NewString(prefix), bare
	}
	if strings.Contains(name, "::") {
		return nil, "", fmt.Errorf("chat: tool name %q contains '::'", name)
	}
	if isString(namespace) && strings.Contains(namespace.S, "::") {
		return nil, "", fmt.Errorf("chat: tool namespace %q contains '::'", namespace.S)
	}
	if namespace.isNone() {
		return nil, name, nil
	}
	return namespace, name, nil
}

func toolNameForEncoding(tool *Value) (string, error) {
	ns := tool.Get("namespace")
	if ns != nil && ns.Kind == Object {
		ns = ns.Get("name")
	}
	name := tool.Get("name")
	if !isString(name) {
		return "", errors.New("chat: tool without a name")
	}
	n, bare, err := splitToolName(name.S, ns)
	if err != nil {
		return "", err
	}
	if n == nil {
		return bare, nil
	}
	s, err := pyStr(n)
	return s + "::" + bare, err
}

func toolsFromOpenAI(tools []*Value) ([]*Value, error) {
	var out []*Value
	for _, t := range tools {
		f := t.Get("function")
		if f == nil || f.Kind != Object {
			return nil, errors.New("chat: tool without a function")
		}
		f = f.ShallowClone()
		if ns := t.Get("namespace"); !ns.isNone() {
			f.Set("namespace", ns)
		}
		name, err := toolNameForEncoding(f)
		if err != nil {
			return nil, err
		}
		f.Set("name", NewString(name))
		ns := f.Delete("namespace")
		if ns != nil && ns.Kind == Object && ns.Get("description").Truthy() {
			d, err := pyStr(ns.Get("description"))
			if err != nil {
				return nil, err
			}
			fd := ""
			if x := f.Get("description"); x.Truthy() {
				if fd, err = pyStr(x); err != nil {
					return nil, err
				}
			}
			f.Set("description", NewString(d+"\n"+fd))
		}
		out = append(out, f)
	}
	return out, nil
}

func toolCallsFromOpenAI(calls []*Value) ([]*Value, error) {
	var out []*Value
	for _, tc := range calls {
		f := tc.Get("function")
		ns := tc.Get("namespace")
		if !ns.Truthy() {
			ns = f.Get("namespace")
		}
		name := f.Get("name")
		if !isString(name) {
			return nil, errors.New("chat: tool call without a name")
		}
		n, bare, err := splitToolName(name.S, ns)
		if err != nil {
			return nil, err
		}
		if !f.Has("arguments") {
			return nil, errors.New("chat: tool call without arguments")
		}
		c := NewObject()
		c.Set("name", NewString(bare))
		c.Set("arguments", f.Get("arguments"))
		if n != nil {
			c.Set("namespace", n)
		}
		out = append(out, c)
	}
	return out, nil
}

func renderTools(tools []*Value) string {
	var schemas []string
	for _, t := range tools {
		schemas = append(schemas, t.Dumps())
	}
	r := strings.NewReplacer("{tool_schemas}", strings.Join(schemas, "\n"), "{dsml_token}", dsmlToken,
		"{tc_block_name}", toolCallsBlockName, "{tool_call_tag_name}", toolCallTagName,
		"{tool_parameter_tag_name}", toolParameterTagName, "{thinking_start_token}", thinkingStartToken,
		"{thinking_end_token}", thinkingEndToken)
	return r.Replace(toolsTemplate)
}

// encodeArgumentsDSML renders a call's arguments as DSML parameters. JSON
// strings are decoded (twice, if doubly encoded); anything that is not an
// object becomes one parameter named "arguments".
func encodeArgumentsDSML(call *Value) (string, error) {
	args := call.Get("arguments")
	orig := args
	if args == nil || args.Kind != Object {
		for range 2 {
			if !isString(args) {
				break
			}
			v, err := ParseJSON(args.S)
			if err != nil {
				break
			}
			args = v
		}
		if args == nil || args.Kind != Object {
			args = NewObject()
			if orig == nil {
				orig = &Value{Kind: Null}
			}
			args.Set("arguments", orig)
		}
	}
	var parts []string
	for _, k := range args.Keys() {
		v := args.Get(k)
		isStr, val := "false", v.Dumps()
		if isString(v) {
			isStr, val = "true", v.S
		}
		parts = append(parts, fmt.Sprintf("<%s%s name=\"%s\" string=\"%s\">%s</%s%s>", dsmlToken, toolParameterTagName, k, isStr, val, dsmlToken, toolParameterTagName))
	}
	return strings.Join(parts, "\n"), nil
}

// renderMessage renders messages[index] (render_message).
func renderMessage(index int, messages []*Value, mode string, dropThinking bool, effort *Value) (string, error) {
	m := messages[index]
	last := lastUserIndex(messages)
	role := m.Get("role")
	content := m.Get("content")
	var tools, calls []*Value
	var err error
	if t := m.Get("tools"); t.Truthy() {
		if t.Kind != Array {
			return "", errors.New("chat: tools must be a list")
		}
		if tools, err = toolsFromOpenAI(t.A); err != nil {
			return "", err
		}
	}
	if c := m.Get("tool_calls"); c.Truthy() {
		if c.Kind != Array {
			return "", errors.New("chat: tool_calls must be a list")
		}
		if calls, err = toolCallsFromOpenAI(c.A); err != nil {
			return "", err
		}
	}
	effortPrompt, err := renderReasoningEffort(index, mode, effort)
	if err != nil {
		return "", err
	}
	var p strings.Builder
	if index == 0 && (effortPrompt != "" || roleIs(m, "system")) {
		p.WriteString(systemSP)
	}
	p.WriteString(effortPrompt)

	orEmpty := func(v *Value) (string, error) {
		if !v.Truthy() {
			return "", nil
		}
		return pyStr(v)
	}
	switch {
	case !isString(role):
		return "", errors.New("chat: message without a role")
	case role.S == "system":
		if index > 0 {
			p.WriteString(systemSP)
		}
		c, err := orEmpty(content)
		if err != nil {
			return "", err
		}
		p.WriteString(c)
		if len(tools) > 0 {
			p.WriteString("\n\n" + renderTools(tools))
		}
		if rf := m.Get("response_format"); rf.Truthy() {
			p.WriteString("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" + rf.Dumps())
		}
	case role.S == "user":
		p.WriteString(userSP)
		if blocks := m.Get("content_blocks"); blocks.Truthy() {
			var parts []string
			for _, b := range blocks.A {
				t := b.Get("type")
				switch {
				case isString(t) && t.S == "text":
					s, err := blockText(b)
					if err != nil {
						return "", err
					}
					parts = append(parts, s)
				case isString(t) && t.S == "tool_result":
					tc := b.Get("content")
					if !b.Has("content") {
						tc = NewString("")
					}
					var s string
					if tc != nil && tc.Kind == Array {
						var texts []string
						for _, x := range tc.A {
							if xt := x.Get("type"); isString(xt) && xt.S == "text" {
								s, err := blockText(x)
								if err != nil {
									return "", err
								}
								texts = append(texts, s)
							} else {
								ts, _ := pyStr(xt)
								texts = append(texts, "[Unsupported "+ts+"]")
							}
						}
						s = strings.Join(texts, "\n\n")
					} else if s, err = pyStr(tc); err != nil {
						return "", err
					}
					parts = append(parts, "<tool_result>"+s+"</tool_result>")
				default:
					ts, _ := pyStr(t)
					parts = append(parts, "[Unsupported "+ts+"]")
				}
			}
			p.WriteString(strings.Join(parts, "\n\n"))
		} else {
			c, err := orEmpty(content)
			if err != nil {
				return "", err
			}
			p.WriteString(c)
		}
	case role.S == "latest_reminder":
		c, err := pyStr(content)
		if err != nil {
			return "", err
		}
		p.WriteString(latestReminderSP + c)
	case role.S == "assistant":
		var tc string
		if len(calls) > 0 {
			var list []string
			for _, c := range calls {
				name, err := toolNameForEncoding(c)
				if err != nil {
					return "", err
				}
				args, err := encodeArgumentsDSML(c)
				if err != nil {
					return "", err
				}
				list = append(list, fmt.Sprintf("<%s%s name=\"%s\">\n%s\n</%s%s>", dsmlToken, toolCallTagName, name, args, dsmlToken, toolCallTagName))
			}
			tc = "\n\n" + fmt.Sprintf("<%s%s>\n%s\n</%s%s>", dsmlToken, toolCallsBlockName, strings.Join(list, "\n"), dsmlToken, toolCallsBlockName)
		}
		summary, err := orEmpty(content)
		if err != nil {
			return "", err
		}
		rc, err := orEmpty(m.Get("reasoning_content"))
		if err != nil {
			return "", err
		}
		prevTask := index > 0 && !messages[index-1].Get("task").isNone()
		thinking := ""
		if mode == "thinking" && !prevTask && (!dropThinking || index > last) {
			thinking = rc + thinkingEndToken
		}
		p.WriteString(thinking + summary + tc)
		if !m.Get("wo_eos").Truthy() {
			p.WriteString(eosToken)
		}
	case role.S == "tool":
		return "", errors.New("chat: tool messages must be merged into user messages")
	default:
		return "", fmt.Errorf("chat: unknown role %q", role.S)
	}

	// What follows decides the transition.
	if index+1 < len(messages) && !roleIs(messages[index+1], "assistant") && !roleIs(messages[index+1], "latest_reminder") {
		return p.String(), nil
	}
	if task := m.Get("task"); !task.isNone() {
		tok, ok := taskTokens[task.S]
		if !isString(task) || !ok {
			return "", fmt.Errorf("chat: invalid task %q", task.S)
		}
		if task.S != "action" {
			p.WriteString(tok)
		} else {
			p.WriteString(assistantSP)
			if mode != "thinking" {
				p.WriteString(thinkingEndToken)
			} else {
				p.WriteString(thinkingStartToken)
			}
			p.WriteString(tok)
		}
	} else if role.S == "user" || role.S == "system" && index > 0 {
		p.WriteString(assistantSP)
		if mode == "thinking" && (!dropThinking || index >= last) {
			p.WriteString(thinkingStartToken)
		} else {
			p.WriteString(thinkingEndToken)
		}
	}
	return p.String(), nil
}

// dropThinkingMessages removes earlier turns' reasoning, and messages
// other than user, system, tool, reminder and search results before the
// last turn.
func dropThinkingMessages(messages []*Value) []*Value {
	last := lastUserIndex(messages)
	var out []*Value
	for i, m := range messages {
		r := m.Get("role")
		keep := isString(r) && slices.Contains([]string{"user", "system", "tool", "latest_reminder", "direct_search_results"}, r.S)
		switch {
		case keep || i >= last:
			out = append(out, m)
		case roleIs(m, "assistant"):
			m = m.ShallowClone()
			m.Delete("reasoning_content")
			out = append(out, m)
		}
	}
	return out
}

func encodeMessagesText(messages []*Value, o ChatOptions, context []*Value) (string, error) {
	full := append(slices.Clone(context), messages...)
	var p strings.Builder
	if !o.NoBOS && len(context) == 0 {
		p.WriteString(bosToken)
	}
	drop := !o.KeepThinking
	for _, m := range full {
		if m.Get("tools").Truthy() {
			drop = false
		}
	}
	var render, ctxLen int
	if o.ThinkingMode == "thinking" && drop {
		full = dropThinkingMessages(full)
		render = len(full) - len(dropThinkingMessages(context))
		ctxLen = len(full) - render
	} else {
		render, ctxLen = len(messages), len(context)
	}
	for i := range render {
		s, err := renderMessage(i+ctxLen, full, o.ThinkingMode, drop, o.ReasoningEffort)
		if err != nil {
			return "", err
		}
		p.WriteString(s)
	}
	return p.String(), nil
}

// ParsedMessage is an assistant message parsed from a completion.
type ParsedMessage struct {
	Content, ReasoningContent string
	// ToolCalls in OpenAI format: {"type": "function", "function": {"name",
	// "arguments"}, "namespace"?}, arguments a JSON string.
	ToolCalls []*Value
}

// readUntilStop returns the text from i to the earliest stop string, the
// index after it and the stop found ("" if none: then the rest of text).
func readUntilStop(i int, text string, stops []string) (int, string, string) {
	at, which := len(text), ""
	for _, s := range stops {
		if p := strings.Index(text[i:], s); p >= 0 && i+p < at {
			at, which = i+p, s
		}
	}
	if which == "" {
		return len(text), text[i:], ""
	}
	return at + len(which), text[i:at], which
}

var (
	// Python's $ also matches before a final newline: \n?\z.
	toolNameRE  = regexp.MustCompile(`(?s)\A\s*name="(.*?)">\n\n?\z`)
	toolParamRE = regexp.MustCompile(`(?s)\A name="(.*?)" string="(true|false)">(.*?)<\n?\z`)
)

func parseToolCalls(i int, text string) (int, string, []*Value, error) {
	callsEnd := "</" + dsmlToken + toolCallsBlockName + ">"
	callStart := "<" + dsmlToken + toolCallTagName
	callEnd := "</" + dsmlToken + toolCallTagName
	paramStart := "<" + dsmlToken + toolParameterTagName
	paramEnd := "/" + dsmlToken + toolParameterTagName
	var calls []*Value
	stop := ""
	for i < len(text) {
		var gap string
		i, gap, stop = readUntilStop(i, text, []string{callStart, callsEnd})
		if gap != ">\n" {
			return 0, "", nil, fmt.Errorf("chat: tool call format: expected '>\\n', got %q", gap)
		}
		if stop == callsEnd {
			break
		}
		if stop == "" {
			return 0, "", nil, errors.New("chat: tool calls end abruptly")
		}
		var nameContent string
		i, nameContent, stop = readUntilStop(i, text, []string{paramStart, callEnd})
		mm := toolNameRE.FindStringSubmatch(nameContent)
		if mm == nil {
			return 0, "", nil, fmt.Errorf("chat: tool name format: %q", nameContent)
		}
		name := mm[1]
		type arg struct{ value, isStr string }
		var order []string
		vals := map[string]arg{}
		for stop == paramStart {
			var pc string
			i, pc, stop = readUntilStop(i, text, []string{paramEnd})
			pm := toolParamRE.FindStringSubmatch(pc)
			if pm == nil {
				return 0, "", nil, fmt.Errorf("chat: parameter format: %q", pc)
			}
			if _, dup := vals[pm[1]]; dup {
				return 0, "", nil, fmt.Errorf("chat: parameter %q given twice", pm[1])
			}
			vals[pm[1]] = arg{pm[3], pm[2]}
			order = append(order, pm[1])
			var after string
			i, after, stop = readUntilStop(i, text, []string{paramStart, callEnd})
			if after != ">\n" {
				return 0, "", nil, fmt.Errorf("chat: parameter format: expected '>\\n', got %q", after)
			}
		}
		var parts []string
		for _, k := range order {
			a := vals[k]
			v := a.value
			if a.isStr == "true" {
				v = NewString(v).Dumps()
			}
			parts = append(parts, NewString(k).Dumps()+": "+v)
		}
		ns, bare, err := splitToolName(name, nil)
		if err != nil {
			return 0, "", nil, err
		}
		c := NewObject()
		c.Set("name", NewString(bare))
		c.Set("arguments", NewString("{"+strings.Join(parts, ", ")+"}"))
		if ns != nil {
			c.Set("namespace", ns)
		}
		calls = append(calls, c)
	}
	return i, stop, calls, nil
}

// ParseCompletion parses a completion (the text after the prompt, through
// the end-of-sentence token) into an assistant message
// (parse_message_from_completion_text). It accepts only well-formed output.
func ParseCompletion(text, mode string) (*ParsedMessage, error) {
	var msg ParsedMessage
	i, stop := 0, ""
	callsStart := "\n\n<" + dsmlToken + toolCallsBlockName
	if mode == "thinking" {
		i, msg.ReasoningContent, stop = readUntilStop(i, text, []string{thinkingEndToken, callsStart})
		if stop != thinkingEndToken {
			return nil, errors.New("chat: completion lacks </think>")
		}
	}
	i, msg.Content, stop = readUntilStop(i, text, []string{eosToken, callsStart})
	if stop == callsStart {
		var calls []*Value
		var err error
		if i, stop, calls, err = parseToolCalls(i, text); err != nil {
			return nil, err
		}
		var rest string
		i, rest, stop = readUntilStop(i, text, []string{eosToken})
		if rest != "" {
			return nil, errors.New("chat: text after the tool calls")
		}
		for _, c := range calls {
			f := NewObject()
			f.Set("name", c.Get("name"))
			f.Set("arguments", c.Get("arguments"))
			o := NewObject()
			o.Set("type", NewString("function"))
			o.Set("function", f)
			if ns := c.Get("namespace"); ns != nil {
				o.Set("namespace", ns)
			}
			msg.ToolCalls = append(msg.ToolCalls, o)
		}
	} else if stop != eosToken {
		return nil, errors.New("chat: completion lacks the end-of-sentence token")
	}
	if i != len(text) || (stop != eosToken && stop != "") {
		return nil, errors.New("chat: text after the end of the message")
	}
	for _, sp := range []string{bosToken, eosToken, thinkingStartToken, thinkingEndToken, dsmlToken} {
		if strings.Contains(msg.Content, sp) || strings.Contains(msg.ReasoningContent, sp) {
			return nil, fmt.Errorf("chat: special token %q in the message", sp)
		}
	}
	return &msg, nil
}
