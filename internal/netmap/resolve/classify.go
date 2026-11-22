package resolve

import (
	"cmp"
	"fmt"
	"net"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// Device classes (spec §12). Leaves are what we want to know; "computer" and
// "mobile" are honest answers when the evidence stops short of a leaf.
const (
	ClassDesktop = "desktop"
	ClassLaptop  = "laptop"
	ClassServer  = "server"
	ClassPhone   = "phone"
	ClassTablet  = "tablet"
	ClassTV      = "tv"
	ClassIoT     = "iot"
	ClassPrinter = "printer"
	ClassNetwork = "network"

	ClassComputer = "computer"
	ClassMobile   = "mobile"
)

var classParent = map[string]string{
	ClassDesktop: ClassComputer, ClassLaptop: ClassComputer, ClassServer: ClassComputer,
	ClassPhone: ClassMobile, ClassTablet: ClassMobile,
}

// Classes lists every class name accepted in a "class" claim.
var Classes = []string{ClassDesktop, ClassLaptop, ClassServer, ClassComputer, ClassPhone, ClassTablet,
	ClassMobile, ClassTV, ClassIoT, ClassPrinter, ClassNetwork}

// Class is what kind of device this is, inferred from its claims. It is
// derived on every build and never stored: better rules improve old maps.
type Class struct {
	Kind    string   `json:"kind"`
	Conf    float64  `json:"conf"`
	Reasons []string `json:"reasons"`
	// Alt is the strongest competing class, when there is one worth showing.
	Alt     string  `json:"alt,omitempty"`
	AltConf float64 `json:"alt_conf,omitempty"`
}

type vote struct {
	kind string
	w    float64
	why  string
	src  string // the rule that fired: one rule is one fact, however many values match it
	val  string // the value it fired on: one value is one fact, however many rules match it
}

// hint is a vote whose reason is filled in where it fires.
type hint struct {
	kind string
	w    float64
}

type rule struct {
	re   *regexp.Regexp
	kind string
	w    float64
}

func rules(kind string, w float64, patterns ...string) []rule {
	out := make([]rule, len(patterns))
	for i, p := range patterns {
		out[i] = rule{regexp.MustCompile(`(?i)` + p), kind, w}
	}
	return out
}

func join(lists ...[]rule) []rule { return slices.Concat(lists...) }

// Apple model identifiers (mDNS _device-info, AirPlay): exact and reliable.
var appleModels = join(
	rules(ClassPhone, 0.95, `^iphone\d`),
	rules(ClassTablet, 0.95, `^ipad\d`),
	rules(ClassMobile, 0.8, `^ipod\d`),
	rules(ClassLaptop, 0.95, `^macbook`),
	rules(ClassDesktop, 0.95, `^imac`, `^macmini`, `^macpro`),
	rules(ClassLaptop, 0.95, macLaptops),
	rules(ClassDesktop, 0.95, macDesktops),
	rules(ClassComputer, 0.9, `^mac\d+,\d`), // a newer Mac: MacBooks and desktops share the scheme
	rules(ClassTV, 0.6, `^appletv3,2$`),     // what third-party AirPlay receivers claim to be
	rules(ClassTV, 0.95, `^appletv([0-24-9]|\d\d|3,1)`),
	rules(ClassIoT, 0.9, `^audioaccessory`, `^watch\d`),
)

// Apple-silicon Mac model identifiers (Mac13,1 …) do not say laptop or
// desktop; these are the ones known when this was written.
const (
	macLaptops  = `^mac(14,(2|5|6|7|9|10|15)|15,(3|6|7|8|9|10|11|12|13)|16,(1|5|6|7|8|12|13))$`
	macDesktops = `^mac(13,(1|2)|14,(3|8|12|13|14)|15,(4|5|14)|16,(2|3|9|10|11))$`
)

// Words in names, models and descriptions.
var nameRules = join(
	rules(ClassPhone, 0.85, `iphone`),
	rules(ClassTablet, 0.85, `ipad`, `galaxy[- _]?tab`, `\bmatepad`, `\bkindle`),
	rules(ClassPhone, 0.75, `\bpixel[- _]?\d`, `galaxy[- _]?(s|a|m|z|note)\d`, `\bredmi`, `\boneplus`, `\bpoco[- _]?[fmx]\d`,
		`\bmoto[- _]?[gez]\d*\b`, `\bxperia`, `\bnokia[- _]?[gcx]?\d`, `\bphone\b`),
	rules(ClassMobile, 0.55, `^android[-_]`, `\bandroid\b`),
	rules(ClassLaptop, 0.85, `macbook`, `^laptop-`, `noutbuk`, `ноутбук`), // Windows names portable machines LAPTOP-XXXX
	rules(ClassLaptop, 0.5, `[-_]nb[-_]`),
	rules(ClassServer, 0.55, `^win-[a-z0-9]{11}\b`), // Windows Server's default host name
	rules(ClassPhone, 0.75, `telefon`, `телефон`, `смартфон`),
	rules(ClassTablet, 0.75, `planshet`, `планшет`),
	rules(ClassTV, 0.75, `televizor`, `телевизор`, `eshare`, `\bsber[- _]?box`, `yandex[- _]?module`),
	rules(ClassPrinter, 0.75, `принтер`, `\bmfu\b`, `мфу`),
	rules(ClassIoT, 0.8, `yandex[- _]?(station|mini|micro|lite|midi|max)`, `yandexio`, `\bsber[- _]?boom`,
		`capsula|капсула`, `\bmarusia`, `колонка`),
	rules(ClassLaptop, 0.75, `thinkpad`, `\blatitude\b`, `elitebook`, `probook`, `zenbook`, `vivobook`, `ideapad`,
		`\bsurface[- _]?(laptop|book|pro|go)`, `chromebook`, `notebook`, `\blaptop\b`, `\bmatebook`, `\bxps[- _]?1[3-7]`),
	rules(ClassDesktop, 0.8, `\bimac\b`, `mac[- _]?mini`, `mac[- _]?studio`, `mac[- _]?pro\b`),
	rules(ClassDesktop, 0.6, `^desktop-`, `\bdesktop\b`, `optiplex`, `thinkcentre`, `prodesk`, `elitedesk`, `\bnuc\d*\b`,
		`\bworkstation\b`, `\bgaming[- _]?pc\b`),
	rules(ClassServer, 0.7, `\bserver\b`, `^srv`, `\bsrv\d*\b`, `proxmox`, `\bpve\d*\b`, `\besxi`, `truenas`, `unraid`,
		`synology`, `diskstation`, `rackstation`, `\bqnap\b`, `\bnas\d*\b`, `poweredge`, `proliant`, `\bidrac`, `\bilo\b`,
		`\bopenmediavault`),
	rules(ClassTV, 0.8, `smart[- _]?tv`, `\btv\b`, `bravia`, `webos`, `tizen`, `\broku`, `chromecast`, `fire[- _]?tv`,
		`android[- _]?tv`, `google[- _]?tv`, `apple[- _]?tv`, `shield`, `\bviera`, `\bkdl-`, `\bqled`, `\boled\b`,
		`mediarenderer`, `playstation`, `\bps[45]\b`, `\bxbox`, `nintendo`, `\bmi[- _]?box`, `\bkodi\b`, `\bplex\b`),
	rules(ClassIoT, 0.75, `esp[-_]?(32|8266)`, `espressif`, `esphome`, `tasmota`, `shelly`, `sonoff`, `\btuya`,
		`smartlife`, `\bhue\b`, `signify`, `\bnest\b`, `\becho\b`, `alexa`, `google[- _]?home`, `homepod`, `sonos`,
		`\bring\b`, `wyze`, `\btapo`, `\bkasa\b`, `meross`, `broadlink`, `yeelight`, `roborock`, `ecovacs`, `\broomba`,
		`ezviz`, `hikvision`, `dahua`, `reolink`, `ip[- _]?cam`, `\bcamera\b`, `thermostat`, `doorbell`, `\bplug\b`,
		`\bbulb\b`, `\bsensor\b`, `zigbee`, `z-?wave`, `\bwemo`, `\blifx`, `nanoleaf`, `\bmiio\b`, `\bzhimi`,
		`\bdreame`, `\bvacuum`, `\bspeaker\b`, `\bsoundbar\b`, `\bbridge\b`, `\bwatch\b`, `\bwled\b`, `\bopenbeken`),
	rules(ClassPrinter, 0.8, `printer`, `laserjet`, `officejet`, `deskjet`, `\benvy[- _]?\d`, `\bpixma`, `imageclass`,
		`imagerunner`, `ecotank`, `\bworkforce\b`, `\bmfc-`, `\bdcp-`, `\bhl-[a-z]?\d`, `ecosys`, `bizhub`, `\bmfp\b`),
	rules(ClassPrinter, 0.55, `^brother`, `kyocera`, `lexmark`, `xerox`, `ricoh`, `^epson`, `konica`),
	rules(ClassNetwork, 0.75, `routeros`, `mikrotik`, `cisco ios`, `\bjunos`, `edgeos`, `edgerouter`, `edgeswitch`,
		`unifi`, `ubiquiti`, `openwrt`, `dd-wrt`, `\barcher[- _]?[a-z]{0,2}\d`, `\bdeco\b`, `keenetic`, `fritz!?box`,
		`procurve`, `arubaos`, `\brt-[a-z]{1,3}\d`, `\brouter\b`, `access[- _]?point`, `\bopnsense`, `pfsense`,
		`\beero\b`, `\bmesh\b`, `\bzyxel`, `\bomada`, `\bwi-?fi[- _]?extender`, `\brepeater\b`),
)

// Keys whose free text goes through nameRules, and how much to trust a
// match there. A device names itself in name/model/hostname; reverse DNS is
// someone else's naming.
var textKeys = map[string]float64{
	"model": 1, "name": 1, "hostname": 1, "mdns.name": 1, "mdns.instance": 0.9, "vendor": 0.8,
	"upnp.device": 1, "upnp.type": 1, "upnp.server": 0.8, "http.server": 0.6, "os.desc": 1, "rdns": 0.5,
	"dhcp.vendor": 0.8,
}

var serviceVotes = map[string]hint{
	"_ipp._tcp": {ClassPrinter, 0.9}, "_ipps._tcp": {ClassPrinter, 0.9}, "_printer._tcp": {ClassPrinter, 0.9},
	"_pdl-datastream._tcp": {ClassPrinter, 0.9}, "_uscan._tcp": {ClassPrinter, 0.7}, "_scanner._tcp": {ClassPrinter, 0.7},
	"_googlecast._tcp": {ClassTV, 0.6}, "_androidtvremote2._tcp": {ClassTV, 0.9}, "_airplay._tcp": {ClassTV, 0.15},
	"_raop._tcp": {ClassIoT, 0.25}, "_spotify-connect._tcp": {ClassIoT, 0.25}, "_sonos._tcp": {ClassIoT, 0.9},
	"_hap._tcp": {ClassIoT, 0.6}, "_homekit._tcp": {ClassIoT, 0.6}, "_matter._tcp": {ClassIoT, 0.8},
	"_matterc._udp": {ClassIoT, 0.8}, "_esphomelib._tcp": {ClassIoT, 0.95}, "_mqtt._tcp": {ClassServer, 0.4},
	"_workstation._tcp": {ClassComputer, 0.6}, "_ssh._tcp": {ClassComputer, 0.4}, "_sftp-ssh._tcp": {ClassComputer, 0.4},
	"_smb._tcp": {ClassComputer, 0.4}, "_afpovertcp._tcp": {ClassComputer, 0.4}, "_rfb._tcp": {ClassComputer, 0.4},
	"_adisk._tcp": {ClassServer, 0.5}, "_nfs._tcp": {ClassServer, 0.5}, "_home-assistant._tcp": {ClassServer, 0.6},
	"_yandexio._tcp": {ClassIoT, 0.85}, "_eshareserver._tcp": {ClassTV, 0.75}, "_viziocast._tcp": {ClassTV, 0.9},
	"_amzn-wplay._tcp": {ClassTV, 0.7}, "_nvstream._tcp": {ClassComputer, 0.4},
}

var portVotes = map[int]hint{
	9100: {ClassPrinter, 0.6}, 515: {ClassPrinter, 0.6}, 631: {ClassPrinter, 0.35},
	3389: {ClassComputer, 0.6}, 5900: {ClassComputer, 0.3}, 139: {ClassComputer, 0.3}, 445: {ClassComputer, 0.3},
	62078: {ClassMobile, 0.9}, // Apple lockdownd: iPhone and iPad only
	8008:  {ClassTV, 0.35}, 8009: {ClassTV, 0.45}, 554: {ClassIoT, 0.5}, 1883: {ClassServer, 0.3},
	23: {ClassNetwork, 0.3}, 53: {ClassNetwork, 0.4},
}

var roleVotes = map[string]hint{
	"gateway": {ClassNetwork, 0.95}, "router": {ClassNetwork, 0.5}, "switch": {ClassNetwork, 0.95},
	"access-point": {ClassNetwork, 0.95}, "dhcp-server": {ClassNetwork, 0.4}, "observer": {ClassComputer, 0.5},
}

// Rules for specific keys, checked in addition to nameRules.
var keyRules = map[string][]rule{
	"upnp.device": join(
		rules(ClassNetwork, 0.9, `internetgatewaydevice`, `wandevice`, `wanconnectiondevice`, `wfadevice`, `wlanaccesspoint`),
		rules(ClassTV, 0.55, `mediarenderer`),
		rules(ClassTV, 0.8, `dial-multiscreen`),
		rules(ClassPrinter, 0.9, `:printer:`),
		rules(ClassServer, 0.4, `mediaserver`),
		rules(ClassIoT, 0.9, `zoneplayer`),
		rules(ClassIoT, 0.8, `binarylight`, `dimmablelight`, `hvac`, `sensormanagement`, `urn:belkin`),
	),
	"dhcp.vendor": join(
		rules(ClassMobile, 0.55, `^android-dhcp-`),
		rules(ClassComputer, 0.6, `^msft 5\.0`),
		rules(ClassIoT, 0.4, `^udhcp`),
		rules(ClassIoT, 0.6, `yealink`, `polycom`, `aastra`, `ip phone`), // desk phones
		rules(ClassNetwork, 0.6, `^ubnt`, `^cisco`, `^aruba`),
	),
	"os.desc": join(
		rules(ClassServer, 0.85, `windows server`, `vmware esxi`),
		rules(ClassComputer, 0.5, `\bwindows\b`),
		rules(ClassServer, 0.5, `linux .*x86_64`, `freebsd`, `ubuntu`, `debian`, `centos`, `red hat`, `rhel`, `rocky`),
		rules(ClassNetwork, 0.8, `\bcisco\b`, `juniper`, `\bswitch\b`, `\brouter\b`, `arubaos`, `dell networking`,
			`huawei versatile`, `hp j\d`, `\bnetgear\b`, `\bd-link\b`, `\btp-link\b`),
	),
	"ssh.banner": join(
		rules(ClassNetwork, 0.3, `dropbear`),
		rules(ClassNetwork, 0.8, `cisco`, `rosssh`, `routeros`),
		rules(ClassServer, 0.4, `openssh.*(ubuntu|debian|raspbian|fedora|freebsd|el\d)`),
		rules(ClassComputer, 0.5, `openssh_for_windows`),
	),
	"http.server": join(
		rules(ClassNetwork, 0.5, `uhttpd`, `routeros`, `micro_httpd`),
		rules(ClassIoT, 0.4, `\bboa\b`, `goahead`, `mini_httpd`, `app-webs`, `hikvision`, `esp`),
		rules(ClassServer, 0.6, `microsoft-iis`),
		rules(ClassServer, 0.25, `nginx`, `apache`, `caddy`),
	),
	"upnp.type": rules(ClassTV, 0.95, `roku:ecp`),
}

// macVotes reads hints from MAC addresses: VM and container prefixes, and
// randomized ones (phones and laptops randomize per network; fixed gear
// does not).
func macVotes(d *Device) []vote {
	var out []vote
	for _, id := range d.Idents {
		k := obs.Kind(id)
		if k != "lmac" && k != "mac" {
			continue
		}
		mac, err := net.ParseMAC(obs.Value(id))
		if err != nil || len(mac) != 6 {
			continue
		}
		m := strings.ToLower(mac.String())
		switch {
		case strings.HasPrefix(m, "52:54:00"), strings.HasPrefix(m, "00:16:3e"), strings.HasPrefix(m, "00:50:56"),
			strings.HasPrefix(m, "00:0c:29"), strings.HasPrefix(m, "08:00:27"):
			out = append(out, vote{kind: ClassServer, w: 0.5, why: "virtual machine MAC " + m})
		case strings.HasPrefix(m, "02:42:"):
			out = append(out, vote{kind: ClassServer, w: 0.4, why: "container MAC " + m})
		case k == "lmac" && obs.IsLocallyAdministered(mac) && !obs.IsVirtualMAC(mac):
			out = append(out, vote{kind: ClassMobile, w: 0.25, why: "randomized MAC"})
		}
	}
	return out
}

func short(s string) string {
	if len(s) > 40 {
		return s[:39] + "…"
	}
	return s
}

func classVotes(d *Device) []vote {
	var out []vote
	add := func(v vote, cv *ClaimValue, scale float64) {
		v.w *= scale * float64(cv.Conf) / 1000
		if v.w > 0 {
			out = append(out, v)
		}
	}
	match := func(rs []rule, k string, cv *ClaimValue, scale float64) {
		for _, r := range rs {
			if r.re.MatchString(cv.Value) {
				add(vote{r.kind, r.w, fmt.Sprintf("%s %q", k, short(cv.Value)), r.re.String(), strings.ToLower(cv.Value)}, cv, scale)
			}
		}
	}
	for k, vs := range d.Claims {
		for _, cv := range vs {
			switch k {
			case "class":
				if slices.Contains(Classes, cv.Value) && cv.Conf <= 500 {
					// only weakly held declarations are mere evidence (see classifyDevice)
					add(vote{kind: cv.Value, w: 0.95, why: "declared class " + cv.Value}, cv, 1)
				}
			case "role":
				if h, ok := roleVotes[cv.Value]; ok {
					add(vote{kind: h.kind, w: h.w, why: "role " + cv.Value}, cv, 1)
				}
			case "service":
				svc, _, _ := strings.Cut(strings.TrimPrefix(cv.Value, "mdns/"), ":")
				if h, ok := serviceVotes[svc]; ok {
					add(vote{kind: h.kind, w: h.w, why: "announces " + svc}, cv, 1)
				}
			case "port":
				if p, err := strconv.Atoi(strings.TrimPrefix(cv.Value, "tcp/")); err == nil {
					if h, ok := portVotes[p]; ok {
						add(vote{kind: h.kind, w: h.w, why: "open " + cv.Value}, cv, 1)
					}
				}
			case "model":
				match(appleModels, k, cv, 1)
			case "os":
				if cv.Value == "linux" || cv.Value == "darwin" || cv.Value == "windows" || cv.Value == "freebsd" {
					add(vote{kind: ClassComputer, w: 0.3, why: "runs " + cv.Value}, cv, 1)
				}
			}
			match(keyRules[k], k, cv, 1)
			if scale := textKeys[k]; scale > 0 {
				match(nameRules, k, cv, scale)
			}
		}
	}
	return append(out, macVotes(d)...)
}

func combine(ws ...float64) float64 {
	p := 1.0
	for _, w := range ws {
		p *= 1 - w
	}
	return 1 - p
}

// classify combines the votes (spec §12): within a class, independent
// evidence accumulates as 1−Π(1−w), with only the strongest vote counted
// per reason; between top-level groups the runner-up discounts the winner.
func classify(d *Device) *Class {
	votes := classVotes(d)
	if len(votes) == 0 {
		return nil
	}
	// Strongest vote per (kind, rule): "iPad" in the mDNS name and in the
	// instance name, or two "Android-N" names, are one fact, not two.
	type key struct{ kind, by string }
	dedupe := func(votes []vote, by func(vote) string) []vote {
		strongest := map[key]vote{}
		for _, v := range votes {
			k := key{v.kind, cmp.Or(by(v), v.why)}
			if old, ok := strongest[k]; !ok || v.w > old.w || (v.w == old.w && v.why < old.why) {
				strongest[k] = v
			}
		}
		out := make([]vote, 0, len(strongest))
		for _, v := range strongest {
			out = append(out, v)
		}
		return out
	}
	votes = dedupe(dedupe(votes, func(v vote) string { return v.val }), func(v vote) string { return v.src })
	byKind := map[string][]vote{}
	for _, v := range votes {
		byKind[v.kind] = append(byKind[v.kind], v)
	}
	score := map[string]float64{}
	for k, vs := range byKind {
		slices.SortFunc(vs, func(a, b vote) int {
			if a.w != b.w {
				if a.w > b.w {
					return -1
				}
				return 1
			}
			return strings.Compare(a.why, b.why)
		})
		ws := make([]float64, len(vs))
		for i, v := range vs {
			ws[i] = v.w
		}
		score[k] = combine(ws...)
	}
	group := func(k string) string {
		if p := classParent[k]; p != "" {
			return p
		}
		return k
	}
	groupScore := map[string]float64{}
	for k, s := range score {
		g := group(k)
		groupScore[g] = combine(groupScore[g], s)
	}
	type gs struct {
		g string
		s float64
	}
	var ranked []gs
	for g, s := range groupScore {
		ranked = append(ranked, gs{g, s})
	}
	slices.SortFunc(ranked, func(a, b gs) int {
		if a.s != b.s {
			if a.s > b.s {
				return -1
			}
			return 1
		}
		return strings.Compare(a.g, b.g)
	})
	win := ranked[0]
	second := 0.0
	if len(ranked) > 1 {
		second = ranked[1].s
	}
	conf := win.s * (1 - second/2)
	if conf < 0.25 {
		return nil
	}
	kind := win.g
	// refine a group to a leaf when one leaf clearly leads
	var l1, l2 float64
	leaf := ""
	for k, s := range score {
		if classParent[k] != win.g {
			continue
		}
		switch {
		case s > l1 || (s == l1 && k < leaf):
			l2, l1, leaf = l1, s, k
		case s > l2:
			l2 = s
		}
	}
	if leaf != "" && l1 >= 0.4 && l1-l2 >= 0.2 {
		kind = leaf
		conf = min(conf, l1*(1-l2/2))
	}
	c := &Class{Kind: kind, Conf: round2(conf)}
	var reasons []vote
	for k, vs := range byKind {
		if group(k) == win.g && (k == kind || k == win.g || kind == win.g) {
			reasons = append(reasons, vs...)
		}
	}
	slices.SortFunc(reasons, func(a, b vote) int {
		if a.w != b.w {
			if a.w > b.w {
				return -1
			}
			return 1
		}
		return strings.Compare(a.why, b.why)
	})
	for _, v := range reasons {
		if len(c.Reasons) == 4 {
			break
		}
		if !slices.Contains(c.Reasons, v.why) {
			c.Reasons = append(c.Reasons, v.why)
		}
	}
	if len(ranked) > 1 && second >= 0.3 {
		c.Alt, c.AltConf = ranked[1].g, round2(second)
	}
	return c
}

func round2(f float64) float64 { return float64(int(f*100+0.5)) / 100 }

// declared returns the best "class" claim: set by hand, or the observer's
// own chassis. Values are sorted best-first (confidence, then recency).
func declared(d *Device) *ClaimValue {
	for _, cv := range d.Claims["class"] {
		if slices.Contains(Classes, cv.Value) && cv.Conf > 500 {
			return cv
		}
	}
	return nil
}

func classifyAll(g *Graph) {
	for _, d := range g.Devices {
		d.Class = classifyDevice(d)
	}
}

// classifyDevice is classify with a declared class taking precedence: what a
// person (or the firmware about its own chassis) says outweighs guesses. The
// guess is kept as the alternative when it disagrees.
func classifyDevice(d *Device) *Class {
	inferred := classify(d)
	cv := declared(d)
	if cv == nil {
		return inferred
	}
	c := &Class{Kind: cv.Value, Conf: round2(float64(cv.Conf) / 1000), Reasons: []string{"declared class " + cv.Value}}
	if inferred == nil {
		return c
	}
	group := func(k string) string { return cmp.Or(classParent[k], k) }
	if group(inferred.Kind) == group(c.Kind) {
		for _, r := range inferred.Reasons {
			if len(c.Reasons) < 4 && !slices.Contains(c.Reasons, r) {
				c.Reasons = append(c.Reasons, r)
			}
		}
		c.Conf = round2(combine(c.Conf, inferred.Conf))
		c.Alt, c.AltConf = inferred.Alt, inferred.AltConf
	} else {
		c.Alt, c.AltConf = inferred.Kind, inferred.Conf
	}
	return c
}
