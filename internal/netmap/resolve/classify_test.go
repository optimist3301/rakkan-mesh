package resolve

import (
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

func dev(idents []string, kv ...string) *Device {
	d := &Device{Idents: idents, Claims: map[string][]*ClaimValue{}}
	for i := 0; i+1 < len(kv); i += 2 {
		d.Claims[kv[i]] = append(d.Claims[kv[i]], &ClaimValue{Value: kv[i+1], Conf: 1000})
	}
	return d
}

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		d    *Device
		want string // "" = unknown
	}{
		{"iphone by model", dev(nil, "model", "iPhone15,2"), ClassPhone},
		{"iphone by name", dev(nil, "mdns.name", "iPhone-Ivan.local"), ClassPhone},
		{"ipad", dev(nil, "mdns.name", "iPad-Natala-2.local", "mdns.instance", "iPad (Наталья)"), ClassTablet},
		{"macbook by apple-silicon id", dev(nil, "model", "Mac14,2"), ClassLaptop},
		{"mac studio by id", dev(nil, "model", "Mac13,1"), ClassDesktop},
		{"unknown future mac", dev(nil, "model", "Mac99,1"), ClassComputer},
		{"windows laptop default name", dev(nil, "hostname", "LAPTOP-HP6F3TFD"), ClassLaptop},
		{"windows desktop", dev(nil, "hostname", "DESKTOP-4K2J9QX", "dhcp.vendor", "MSFT 5.0"), ClassDesktop},
		{"russian transliteration", dev(nil, "mdns.name", "Noutbuk-Daria.local"), ClassLaptop},
		{"android phone hostname is only mobile", dev(nil, "hostname", "Android_5JA8W566"), ClassMobile},
		{"pixel", dev(nil, "name", "Pixel 8 Pro"), ClassPhone},
		{"apple tv", dev(nil, "model", "AppleTV11,1", "service", "mdns/_airplay._tcp:7000"), ClassTV},
		{"chromecast", dev(nil, "service", "mdns/_googlecast._tcp:8009", "model", "Chromecast"), ClassTV},
		{"android tv", dev(nil, "service", "mdns/_androidtvremote2._tcp:6466"), ClassTV},
		{"dial renderer", dev(nil, "upnp.device", "urn:dial-multiscreen-org:device:dial:1"), ClassTV},
		{"homepod", dev(nil, "model", "AudioAccessory5,1", "role", "router"), ClassIoT},
		{"esphome", dev(nil, "service", "mdns/_esphomelib._tcp:6053"), ClassIoT},
		{"yandex station", dev(nil, "service", "mdns/_yandexio._tcp:1961"), ClassIoT},
		{"shelly by dhcp", dev(nil, "hostname", "shelly1pm-84CCA8", "dhcp.vendor", "udhcp 1.30.1"), ClassIoT},
		{"printer", dev(nil, "service", "mdns/_ipp._tcp:631", "model", "HP LaserJet M404"), ClassPrinter},
		{"printer by port", dev(nil, "port", "tcp/9100", "port", "tcp/515"), ClassPrinter},
		{"gateway", dev(nil, "role", "gateway"), ClassNetwork},
		{"access point", dev(nil, "role", "access-point"), ClassNetwork},
		{"snmp switch", dev(nil, "os.desc", "Cisco IOS Software, C2960 Software"), ClassNetwork},
		{"synology", dev(nil, "mdns.name", "DiskStation.local", "service", "mdns/_smb._tcp:445"), ClassServer},
		{"windows server", dev(nil, "os.desc", "Hardware: Intel64 - Software: Windows Server 2019"), ClassServer},
		{"kvm guest", dev([]string{"lmac:nx/52:54:00:12:34:56"}, "port", "tcp/22"), ClassServer},
		{"declared overrides weak hints", dev(nil, "class", "tv", "port", "tcp/22"), ClassTV},
		{"iphone by lockdownd port", dev(nil, "port", "tcp/62078"), ClassMobile},
		{"nothing telling", dev(nil, "port", "tcp/80", "seen.l2", "neigh"), ""},
		{"wifi ssid is not the device's name", dev(nil, "wifi.ssid", "Living Room TV"), ""},
		{"invalid declared class ignored", dev(nil, "class", "toaster"), ""},
	}
	for _, c := range cases {
		got := classifyDevice(c.d)
		kind := ""
		if got != nil {
			kind = got.Kind
		}
		if kind != c.want {
			t.Errorf("%s: got %q %+v, want %q", c.name, kind, got, c.want)
		}
	}
}

// A declaration outweighs inference; the disagreeing guess stays visible.
func TestClassifyDeclaredWins(t *testing.T) {
	c := classifyDevice(dev(nil, "class", "printer", "role", "gateway"))
	if c.Kind != ClassPrinter || c.Alt != ClassNetwork || c.Conf != 1 {
		t.Fatalf("declared printer on a gateway: %+v", c)
	}
	agree := classifyDevice(dev(nil, "class", "laptop", "role", "observer", "hostname", "thinkpad-x1"))
	if agree.Kind != ClassLaptop || agree.Alt != "" || len(agree.Reasons) < 2 {
		t.Fatalf("agreeing evidence should support the declaration: %+v", agree)
	}
	// an untrusted key's declaration (confidence halved) is only a hint
	weak := dev(nil, "class", "tv", "role", "gateway")
	weak.Claims["class"][0].Conf = 450
	if c := classifyDevice(weak); c.Kind != ClassNetwork {
		t.Fatalf("weak declaration overrode a gateway: %+v", c)
	}
}

// One fact counts once: the same word in several claims, or several rules
// matching one value, must not inflate confidence.
func TestClassifyNoDoubleCounting(t *testing.T) {
	once := classify(dev(nil, "mdns.name", "Android-2.local"))
	many := classify(dev(nil, "mdns.name", "Android-2.local", "mdns.name", "Android-6.local", "hostname", "android-6"))
	if once == nil || many == nil || once.Conf != many.Conf {
		t.Fatalf("repetition changed confidence: %+v vs %+v", once, many)
	}
	both := classify(dev(nil, "mdns.name", "iPhone-Ivan.local", "port", "tcp/62078"))
	if both.Kind != ClassPhone || both.Conf <= once.Conf {
		t.Fatalf("independent evidence should add up: %+v", both)
	}
}

// Conflicting evidence lowers confidence and names the alternative.
func TestClassifyConflict(t *testing.T) {
	c := classify(dev(nil, "mdns.name", "iPhone-iMac.local", "mdns.instance", "iMac"))
	if c == nil || c.Alt == "" || c.Conf > 0.6 {
		t.Fatalf("conflict not reflected: %+v", c)
	}
}

// Untrusted evidence weighs less: claim confidence is already scaled by trust.
func TestClassifyTrustScaled(t *testing.T) {
	d := dev(nil, "role", "gateway")
	full := classify(d).Conf
	d.Claims["role"][0].Conf = 500
	if half := classify(d); half == nil || half.Conf >= full {
		t.Fatalf("half-trusted gateway claim: %v vs %v", half, full)
	}
}

func TestChassisClassInBuild(t *testing.T) {
	b := &builder{t: t}
	b.add(key(1), obs.ModeLocal, "self", 0, []string{"rkm:aa"}, obs.Claim{K: "role", V: "observer"},
		obs.Claim{K: "class", V: "laptop", C: 900})
	g := Build(b.list)
	if d := byIdent(t, g, "rkm:aa"); d.Class == nil || d.Class.Kind != ClassLaptop {
		t.Fatalf("observer class: %+v", d.Class)
	}
}
