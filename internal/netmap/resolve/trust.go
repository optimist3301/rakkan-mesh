package resolve

import (
	"cmp"
	"encoding/hex"
	"net/netip"
	"slices"
	"strconv"
	"strings"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

// Options tune resolution for one viewer. The observation set stays the
// same; only how much each signer is believed changes (spec §10).
type Options struct {
	// Trust is the viewer's book. Nil means every observer is trusted: pure
	// set semantics, used for tests and for inspecting a file as-is.
	Trust *trust.Book
	// ASN maps a public address to its autonomous system (reference data,
	// e.g. an imported ip2asn table). Nil: no AS annotation.
	ASN func(netip.Addr) (AS, bool)
}

// UnknownWeight scales the confidence of claims by observers the viewer
// has not marked as trusted.
var UnknownWeight = 0.5

// Guests of a network, who proved work but hold no certificate, start lower
// than unknown observers and earn their way up: every distinct fact of
// theirs the trusted observers confirm moves them toward UnknownWeight, and
// none ever makes them trusted: only the viewer's book does. A fact the
// trusted observers contradict costs weight in proportion to the share of
// contradictions; an observer with at least MinChecked facts checked, half
// of them contradicted, is left out of the map altogether.
var (
	GuestWeight   = 0.25
	AgreeFull     = 10  // confirmed facts that lift a guest to UnknownWeight
	MinChecked    = 3   // facts compared before an observer can be left out
	DisputedRatio = 0.5 // contradicted share at which it is
	minWeight     = 0.02
)

// Check is how one untrusted observer fared against the trusted ones.
type Check struct {
	Agree    int // distinct (device, claim, value) the trusted observers also state
	Conflict int // ... that they contradict, plus identity merges they refuse
}

func (c *Check) ratio() float64 {
	if c == nil || c.Agree+c.Conflict == 0 {
		return 0
	}
	return float64(c.Conflict) / float64(c.Agree+c.Conflict)
}

func (c *Check) disputed() bool {
	return c != nil && c.Agree+c.Conflict >= MinChecked && c.ratio() >= DisputedRatio
}

// checkedClaims are the claims that have one true value at a time, so that
// two observers stating different ones contradict each other.
var checkedClaims = map[string]bool{
	"hostname": true, "vendor": true, "os": true, "os.desc": true,
	"dhcp.vendor": true, "upnp.server": true, "upnp.client": true,
}

func normClaim(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// crossCheck compares what observers of unknown standing state about each
// device with what the trusted observers state about it. A value the
// trusted never gave, for a claim they did give, is a contradiction (they
// may well disagree among themselves over time, so every value any of
// them gave counts as confirmed). owner maps each observation to its
// device; disputes are the identity merges the trusted evidence refused.
func crossCheck(list []*obs.Observation, owner []string, tc *trustCtx, disputes map[string]map[string]bool) map[string]*Check {
	type slot struct{ dev, claim string }
	given := map[slot]map[string]bool{}
	for i, o := range list {
		if owner[i] == "" || tc.of(o) < trust.Trusted {
			continue
		}
		for _, c := range o.Claims {
			if !checkedClaims[c.K] {
				continue
			}
			s := slot{owner[i], c.K}
			if given[s] == nil {
				given[s] = map[string]bool{}
			}
			given[s][normClaim(c.V)] = true
		}
	}
	out := map[string]*Check{}
	counted := map[string]bool{}
	entry := func(k string) *Check {
		if out[k] == nil {
			out[k] = &Check{}
		}
		return out[k]
	}
	for i, o := range list {
		k := hex.EncodeToString(o.Observer)
		if owner[i] == "" || tc.level[k] != trust.Unknown {
			continue
		}
		for _, c := range o.Claims {
			vals := given[slot{owner[i], c.K}]
			if !checkedClaims[c.K] || len(vals) == 0 {
				continue
			}
			id := k + "\x00" + owner[i] + "\x00" + c.K + "\x00" + normClaim(c.V)
			if counted[id] {
				continue
			}
			counted[id] = true
			if vals[normClaim(c.V)] {
				entry(k).Agree++
			} else {
				entry(k).Conflict++
			}
		}
	}
	for k, set := range disputes {
		if tc.level[k] == trust.Unknown && len(set) > 0 {
			entry(k).Conflict += len(set)
		}
	}
	return out
}

// ObserverInfo describes one signer of the set as seen by this viewer.
type ObserverInfo struct {
	Key          string      `json:"key"`  // short form, as in Device.Observers
	Full         string      `json:"full"` // hex public key
	Name         string      `json:"name,omitempty"`
	Level        trust.Level `json:"level"`
	InheritedVia string      `json:"inherited_via,omitempty"` // key this one was rotated from
	RevokedAt    int64       `json:"revoked_at,omitempty"`
	Successor    string      `json:"successor,omitempty"`
	Observations int         `json:"observations"`
	Ignored      int         `json:"ignored,omitempty"` // blocked, disputed, or signed after revocation
	// Guest: reached the viewer only through a guest of the network.
	Guest bool `json:"guest,omitempty"`
	// Agree and Conflict: facts of this observer that the trusted
	// observers confirmed or contradicted. Weight is what the viewer gives
	// its claims after that (1 for trusted); Disputed leaves it out.
	Agree    int     `json:"agree,omitempty"`
	Conflict int     `json:"conflict,omitempty"`
	Weight   float64 `json:"weight"`
	Disputed bool    `json:"disputed,omitempty"`
}

type trustCtx struct {
	level   map[string]trust.Level // full hex → effective level
	weights map[string]float64     // full hex → weight of its claims
}

func (t *trustCtx) of(o *obs.Observation) trust.Level { return t.level[hex.EncodeToString(o.Observer)] }

func (t *trustCtx) weight(o *obs.Observation) float64 {
	return t.weights[hex.EncodeToString(o.Observer)]
}

// observerWeight is the weight of an observer's claims: full for trusted
// ones, lower for the rest, lower still for guests, and reduced by the
// share of its checked facts that were contradicted.
func observerWeight(l trust.Level, guest bool, c *Check) float64 {
	if l >= trust.Trusted {
		return 1
	}
	w := UnknownWeight
	if guest {
		earned := 1.0
		if AgreeFull > 0 && c != nil {
			earned = min(1, float64(c.Agree)/float64(AgreeFull))
		} else if AgreeFull > 0 {
			earned = 0
		}
		w = GuestWeight + (UnknownWeight-GuestWeight)*earned
	}
	r := c.ratio()
	return max(w*(1-r)*(1-r), minWeight)
}

// Revocation and rotation are observations like any other, so they travel
// with merges. They are honoured only when signed by the key they talk
// about: a key can retire itself, never another key. After a theft the
// owner should revoke without naming a successor (rakkan key rotate
// --compromised-since) and have peers trust the new key explicitly.
//
//	method "revoke", subject ["rkm:<own key>"]
//	key.revoked   = unix ms from which the key's signatures are void
//	key.successor = "rkm:<new key>" that continues this observer
func keyEvents(list []*obs.Observation) (revoked map[string]int64, successor map[string]string) {
	revoked, successor = map[string]int64{}, map[string]string{}
	ambiguous := map[string]bool{}
	for _, o := range list {
		if o.V != obs.SchemaVersion || o.Method != "revoke" {
			continue
		}
		self := hex.EncodeToString(o.Observer)
		if !slices.Contains(o.Subject, "rkm:"+self) {
			continue
		}
		for _, c := range o.Claims {
			switch c.K {
			case "key.revoked":
				at, err := strconv.ParseInt(c.V, 10, 64)
				if err != nil || at <= 0 {
					continue
				}
				if prev, ok := revoked[self]; !ok || at < prev {
					revoked[self] = at
				}
			case "key.successor":
				next := strings.TrimPrefix(c.V, "rkm:")
				if _, err := hex.DecodeString(next); err == nil && len(next) == 64 && next != self {
					if prev, ok := successor[self]; ok && prev != next {
						ambiguous[self] = true
					}
					successor[self] = next
				}
			}
		}
	}
	// Two different successors mean someone else holds the key too (a thief
	// naming their own): nobody inherits anything from it.
	for k := range ambiguous {
		delete(successor, k)
	}
	return revoked, successor
}

// prepare applies the viewer's trust to the set: drops what must not count
// and computes each observer's effective level.
func prepare(list []*obs.Observation, opts Options, rep map[string]*Check) ([]*obs.Observation, *trustCtx, []ObserverInfo) {
	revoked, successor := keyEvents(list)
	counts := map[string]int{}
	for _, o := range list {
		counts[hex.EncodeToString(o.Observer)]++
	}
	t := &trustCtx{level: map[string]trust.Level{}, weights: map[string]float64{}}
	inherited := map[string]string{}
	base := func(k string) trust.Level {
		if opts.Trust == nil {
			return trust.Trusted
		}
		return opts.Trust.Level(k)
	}
	for k := range counts {
		t.level[k] = base(k)
	}
	// A rotated key inherits the trust of the key that named it, unless the
	// viewer decided about it explicitly. Chains are followed to a fixpoint.
	for changed := true; changed; {
		changed = false
		for _, old := range sortedKeys(successor) {
			next := successor[old]
			if _, explicit := opts.Trust.Get(next); explicit || opts.Trust == nil {
				continue
			}
			lo := t.level[old]
			if _, ok := counts[old]; !ok {
				lo = base(old)
			}
			if lo >= trust.Trusted && t.level[next] < lo {
				t.level[next] = lo
				inherited[next] = old
				changed = true
			}
		}
	}

	for k := range counts {
		e, _ := opts.Trust.Get(k)
		t.weights[k] = observerWeight(t.level[k], e.Guest, rep[k])
	}

	kept := make([]*obs.Observation, 0, len(list))
	ignored := map[string]int{}
	for _, o := range list {
		k := hex.EncodeToString(o.Observer)
		at, isRevoked := revoked[k]
		switch {
		case t.level[k] == trust.Blocked:
			ignored[k]++
		case isRevoked && o.Time > at && o.Method != "revoke":
			ignored[k]++
		case t.level[k] < trust.Trusted && rep[k].disputed() && o.Method != "revoke":
			ignored[k]++
		default:
			kept = append(kept, o)
		}
	}

	var infos []ObserverInfo
	for k, n := range counts {
		info := ObserverInfo{Key: k[:12], Full: k, Level: t.level[k], Observations: n, Ignored: ignored[k],
			RevokedAt: revoked[k], Successor: successor[k], Weight: t.weights[k]}
		if e, ok := opts.Trust.Get(k); ok {
			info.Name = e.Name
			info.Guest = e.Guest && t.level[k] == trust.Unknown
		}
		if c := rep[k]; c != nil {
			info.Agree, info.Conflict, info.Disputed = c.Agree, c.Conflict, c.disputed()
		}
		if v := inherited[k]; v != "" {
			info.InheritedVia = v
		}
		infos = append(infos, info)
	}
	slices.SortFunc(infos, func(a, b ObserverInfo) int {
		return cmp.Or(cmp.Compare(b.Level, a.Level), cmp.Compare(b.Observations, a.Observations), strings.Compare(a.Full, b.Full))
	})
	return kept, t, infos
}
