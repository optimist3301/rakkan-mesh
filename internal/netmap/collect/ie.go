//go:build linux

package collect

import (
	"strconv"
	"strings"

	"github.com/mdlayher/wifi"
)

// securityOf names the authentication of a BSS from its RSN AKM suites.
// Without an RSN element the network is open or uses legacy WEP/WPA, which
// the scan cache does not distinguish.
func securityOf(r wifi.RSNInfo) string {
	if !r.IsInitialized() {
		return "open/legacy"
	}
	var names []string
	add := func(s string) {
		for _, n := range names {
			if n == s {
				return
			}
		}
		names = append(names, s)
	}
	for _, a := range r.AKMs {
		switch a {
		case wifi.RSNAkm8021X, wifi.RSNAkmFT8021X, wifi.RSNAkm8021XSHA256,
			wifi.RSNAkm8021XSuiteB, wifi.RSNAkm8021XCNSA, wifi.RSNAkmFT8021XSHA384:
			add("WPA2-EAP")
		case wifi.RSNAkmPSK, wifi.RSNAkmFTPSK, wifi.RSNAkmPSKSHA256:
			add("WPA2-PSK")
		case wifi.RSNAkmSAE, wifi.RSNAkmFTSAE, 0x000FAC18, 0x000FAC19: // SAE, FT-SAE, SAE-EXT-KEY
			add("WPA3-SAE")
		case 0x000FAC12:
			add("OWE")
		default:
			add("RSN-" + strconv.FormatUint(uint64(a)&0xff, 10))
		}
	}
	if len(names) == 0 {
		return "RSN"
	}
	return strings.Join(names, "/")
}

// sanitize keeps printable text from untrusted radio data.
func sanitize(b []byte) string {
	s := strings.ToValidUTF8(string(b), "")
	s = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, s)
	return strings.TrimSpace(s)
}

// freqChannel converts a centre frequency in MHz to (band, channel).
func freqChannel(mhz int) (string, int) {
	switch {
	case mhz == 2484:
		return "2.4GHz", 14
	case mhz >= 2412 && mhz <= 2472:
		return "2.4GHz", (mhz - 2407) / 5
	case mhz >= 5955 && mhz <= 7115:
		return "6GHz", (mhz - 5950) / 5
	case mhz >= 5000 && mhz <= 5900:
		return "5GHz", (mhz - 5000) / 5
	}
	return "", 0
}

// quantizeRSSI floors to 3 dB so that passive aggregation folds the normal
// jitter of a stationary measurement into one observation.
func quantizeRSSI(dbm int) string {
	q := dbm / 3
	if dbm%3 != 0 && dbm < 0 {
		q-- // floor, not truncation toward zero
	}
	return strconv.Itoa(q * 3)
}
