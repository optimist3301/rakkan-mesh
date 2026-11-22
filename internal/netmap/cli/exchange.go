//go:build linux

package cli

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/resolve"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/store"
	"github.com/optimist3301/rakkan-mesh/internal/netmap/trust"
)

// parseMixed parses flags that may come before, between or after
// positional arguments, and returns the positionals in order.
func parseMixed(fs *flag.FlagSet, args []string) []string {
	var pos []string
	for {
		fs.Parse(args)
		if fs.NArg() == 0 {
			return pos
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
}

// ---------- trust ----------

func cmdTrust(args []string) error {
	const help = `usage: rakkan-node map trust [list]
       rakkan-node map trust set KEY|NAME LEVEL [--name NAME]   LEVEL: blocked, unknown, trusted
       rakkan-node map trust rm KEY|NAME

KEY is a hex prefix (≥6 characters) of an observer key, as listed. The book
is local: it decides how much you believe each signer and who may sync with
you; it never travels with a map.
  blocked  observations ignored, never re-shared
  unknown  default: claims count half and may not merge identities that
           trusted evidence keeps apart
  trusted  full weight; may connect to "rakkan-node map sync serve"
`
	var g globals
	fs := flag.NewFlagSet("trust", flag.ExitOnError)
	g.register(fs)
	name := fs.String("name", "", "a name for the key")
	pos := parseMixed(fs, args)
	sub, target, level := "list", "", ""
	if len(pos) > 0 {
		sub = pos[0]
	}
	if len(pos) > 1 {
		target = pos[1]
	}
	if len(pos) > 2 {
		level = pos[2]
	}

	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	book, err := st.TrustBook()
	if err != nil {
		return err
	}
	known, err := st.Observers()
	if err != nil {
		return err
	}
	own, _, err := obs.LoadOrCreateKey(g.key)
	if err != nil {
		return err
	}
	ownHex := hex.EncodeToString(own.Public().(ed25519.PublicKey))

	switch sub {
	case "list":
		list, _ := st.All()
		gr := resolve.BuildWith(list, resolve.Options{Trust: book.WithSelf(own.Public().(ed25519.PublicKey))})
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "KEY\tLEVEL\tNAME\tOBSERVATIONS\tNOTE")
		listed := map[string]bool{}
		for _, i := range gr.ObserverInfo {
			listed[i.Full] = true
			note := ""
			if i.InheritedVia != "" {
				note += "trust inherited from rotated key " + i.InheritedVia[:12] + "; "
			}
			if i.RevokedAt > 0 {
				note += "revoked from " + time.UnixMilli(i.RevokedAt).Format(time.DateTime) + "; "
			}
			if i.Successor != "" {
				note += "succeeded by " + i.Successor[:12] + "; "
			}
			if i.Guest {
				note += "guest of the network; "
			}
			if i.Agree+i.Conflict > 0 {
				note += fmt.Sprintf("%d facts confirmed by trusted observers, %d contradicted; ", i.Agree, i.Conflict)
			}
			if i.Level < trust.Trusted && i.Level != trust.Blocked {
				note += fmt.Sprintf("weight %.2f; ", i.Weight)
			}
			if i.Disputed {
				note += "DISPUTED, left out of the map; "
			}
			if i.Ignored > 0 {
				note += strconv.Itoa(i.Ignored) + " ignored"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\n", i.Full[:16], i.Level, i.Name, i.Observations, note)
		}
		for _, e := range book.Entries() {
			if !listed[e.Key] {
				fmt.Fprintf(tw, "%s\t%s\t%s\t0\tno observations yet\n", e.Key[:16], e.Level, e.Name)
			}
		}
		return tw.Flush()
	case "set", "rm":
		if target == "" {
			fmt.Print(help)
			return errors.New("missing key")
		}
		k, err := trust.Find(target, book, append(known, ownHex))
		if err != nil {
			return err
		}
		if k == ownHex {
			return errors.New("that is this installation's own key: it is always self")
		}
		if sub == "rm" {
			if err := st.DeleteTrust(k); err != nil {
				return err
			}
			fmt.Printf("%s: back to unknown\n", k[:16])
			return nil
		}
		l, err := trust.ParseLevel(level)
		if err != nil || l == trust.Self {
			fmt.Print(help)
			return fmt.Errorf("level must be blocked, unknown or trusted")
		}
		if err := st.SetTrust(trust.Entry{Key: k, Level: l, Name: *name}); err != nil {
			return err
		}
		fmt.Printf("%s: %s\n", k[:16], l)
		return nil
	}
	fmt.Print(help)
	return fmt.Errorf("unknown trust command %q", sub)
}

// ---------- key ----------

func cmdKey(args []string) error {
	const help = `usage: rakkan-node map key            show this observer's key
       rakkan-node map key rotate     replace the key; the old one names the new one as
                             successor, so peers that trusted it trust the new one
       rakkan-node map key rotate --compromised-since TIME
                             the key was stolen: its signatures from TIME on are
                             void everywhere the revocation reaches; no successor is
                             named (a thief could name theirs), so peers must trust
                             the new key explicitly. TIME: RFC 3339 or 36h / 7d ago

Revocations are observations: export or sync after rotating so they spread.
`
	var g globals
	fs := flag.NewFlagSet("key", flag.ExitOnError)
	g.register(fs)
	compromised := fs.String("compromised-since", "", "the key leaked at this time")
	pos := parseMixed(fs, args)
	sub := ""
	if len(pos) > 0 {
		sub = pos[0]
	}

	old, created, err := obs.LoadOrCreateKey(g.key)
	if err != nil {
		return err
	}
	oldPub := old.Public().(ed25519.PublicKey)
	switch sub {
	case "":
		if created {
			fmt.Fprintf(os.Stderr, "created observer key %s\n", g.key)
		}
		fmt.Printf("%s\n  file %s\n  others trust you with: rakkan-node map trust set %s trusted --name …\n",
			obs.RKMID(oldPub), g.key, hex.EncodeToString(oldPub))
		return nil
	case "rotate":
	default:
		fmt.Print(help)
		return fmt.Errorf("unknown key command %q", sub)
	}

	at := time.Now()
	if *compromised != "" {
		if at, err = parseWhen(*compromised); err != nil {
			return err
		}
	}
	_, newKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	newPub := newKey.Public().(ed25519.PublicKey)
	claims := []obs.Claim{{K: "key.revoked", V: strconv.FormatInt(at.UnixMilli(), 10)}}
	if *compromised == "" {
		claims = append(claims, obs.Claim{K: "key.successor", V: obs.RKMID(newPub)})
	}
	rev, err := obs.Sign(obs.Body{Time: time.Now().UnixMilli(), Mode: obs.ModeLocal, Method: "revoke",
		Subject: []string{obs.RKMID(oldPub)}, Claims: claims}, old)
	if err != nil {
		return err
	}

	st, err := store.Open(g.db)
	if err != nil {
		return err
	}
	defer st.Close()
	if _, err := st.Add([]*obs.Observation{rev}); err != nil {
		return err
	}
	// Keep the old key's past observations counting as ours.
	if err := st.SetTrust(trust.Entry{Key: hex.EncodeToString(oldPub), Level: trust.Self, Name: "previous key"}); err != nil {
		return err
	}
	retired := g.key + ".retired-" + time.Now().Format("20060102-150405")
	if err := os.Rename(g.key, retired); err != nil {
		return err
	}
	if err := obs.SaveKey(g.key, newKey); err != nil {
		os.Rename(retired, g.key) // put the old key back rather than lose both
		return err
	}
	fmt.Printf("new key  %s\nold key  %s revoked from %s, kept as %s\n",
		obs.RKMID(newPub), hex.EncodeToString(oldPub)[:16], at.Format(time.DateTime), retired)
	if *compromised != "" {
		fmt.Println("peers must trust the new key explicitly: rakkan-node map trust set", hex.EncodeToString(newPub)[:12], "trusted")
	}
	fmt.Println("spread the revocation: rakkan-node map publish, or rakkan-node map export -o …")
	return nil
}
