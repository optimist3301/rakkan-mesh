package obs

import (
	"crypto/ed25519"
	"encoding/base32"
	"encoding/hex"
	"net"
	"net/netip"
	"strings"

	"github.com/zeebo/blake3"
)

// Strength of an identifier as evidence of identity (spec §3).
type Strength int

const (
	Weak Strength = iota
	Medium
	Strong
)

var b32 = base32.StdEncoding.WithPadding(base32.NoPadding)

func shortHash(domain string, parts ...string) string {
	h := blake3.New()
	h.Write([]byte(domain))
	for i, p := range parts {
		if i > 0 {
			h.Write([]byte("|"))
		}
		h.Write([]byte(p))
	}
	sum := h.Sum(nil)
	return strings.ToLower(b32.EncodeToString(sum[:10]))
}

// NetID computes the network identifier (spec §4). Independent observers
// in the same network get the same value.
//
// Normally it is derived from the default gateway's MAC and the IPv4 prefix.
// A router-redundancy MAC (VRRP, HSRP, GLBP) identifies a redundancy group,
// not the network: it changes on failover or regrouping while the network
// stays the same, and the same virtual MAC is reused by countless unrelated
// networks. For those the gateway's IP address stands in for the MAC.
func NetID(gwMAC net.HardwareAddr, gwIP netip.Addr, prefix netip.Prefix) string {
	if IsVirtualMAC(gwMAC) && gwIP.IsValid() {
		return "n" + shortHash("rakkan-vnet-v1\x00", gwIP.Unmap().String(), prefix.Masked().String())
	}
	return "n" + shortHash("rakkan-net-v1\x00", strings.ToLower(gwMAC.String()), prefix.Masked().String())
}

// LocalNetID is the fallback when the gateway is unknown. It never collides
// with another observer, which errs on the side of not merging.
func LocalNetID(observer ed25519.PublicKey, iface string, prefix netip.Prefix) string {
	return "x" + shortHash("rakkan-lnet-v1\x00", string(observer), iface, prefix.Masked().String())
}

// HashID derives a short stable id with a one-letter prefix.
func HashID(prefix, s string) string { return prefix + shortHash("rakkan-id-v1\x00", s) }

func Kind(id string) string {
	k, _, _ := strings.Cut(id, ":")
	return k
}

// Scope returns the netid of a scoped identifier ("ip:<net>/1.2.3.4"), or "".
func Scope(id string) string {
	switch Kind(id) {
	case "ip", "rip", "lmac", "host", "mdns":
		_, rest, _ := strings.Cut(id, ":")
		n, _, ok := strings.Cut(rest, "/")
		if ok {
			return n
		}
	case "net":
		return strings.TrimPrefix(id, "net:")
	}
	return ""
}

// Value returns the part of an identifier after kind and scope.
func Value(id string) string {
	_, rest, _ := strings.Cut(id, ":")
	if Scope(id) != "" && Kind(id) != "net" {
		_, v, _ := strings.Cut(rest, "/")
		return v
	}
	return rest
}

func StrengthOf(id string) Strength {
	switch Kind(id) {
	case "rkm", "ssh", "uuid", "lldp", "mac", "lmac", "bssid":
		return Strong
	case "ip", "rip", "gip", "net":
		return Weak
	}
	return Medium
}

func IsLocallyAdministered(mac net.HardwareAddr) bool {
	return len(mac) > 0 && mac[0]&0x02 != 0
}

// IsVirtualMAC reports router-redundancy MACs shared between devices and
// reused across networks: VRRP (00:00:5e:00:01/02:xx), HSRP (00:00:0c:07:ac:xx,
// 00:00:0c:9f:fx:xx), GLBP (00:07:b4:00:...).
func IsVirtualMAC(mac net.HardwareAddr) bool {
	if len(mac) != 6 {
		return false
	}
	switch {
	case mac[0] == 0 && mac[1] == 0 && mac[2] == 0x5e && mac[3] == 0 && (mac[4] == 1 || mac[4] == 2):
		return true
	case mac[0] == 0 && mac[1] == 0 && mac[2] == 0x0c && mac[3] == 0x07 && mac[4] == 0xac:
		return true
	case mac[0] == 0 && mac[1] == 0 && mac[2] == 0x0c && mac[3] == 0x9f && mac[4]&0xf0 == 0xf0:
		return true
	case mac[0] == 0 && mac[1] == 0x07 && mac[2] == 0xb4 && mac[3] == 0:
		return true
	}
	return false
}

func isUsableMAC(mac net.HardwareAddr) bool {
	if len(mac) != 6 || mac[0]&0x01 != 0 { // multicast/broadcast
		return false
	}
	for _, b := range mac {
		if b != 0 {
			return true
		}
	}
	return false
}

// MACID returns the identifier for a MAC seen in network netid: global MACs
// are unscoped, random/virtual ones are scoped to the network. Empty if the
// MAC cannot identify a unicast device.
func MACID(mac net.HardwareAddr, netid string) string {
	if !isUsableMAC(mac) {
		return ""
	}
	m := strings.ToLower(mac.String())
	if IsLocallyAdministered(mac) || IsVirtualMAC(mac) {
		if netid == "" {
			return ""
		}
		return "lmac:" + netid + "/" + m
	}
	return "mac:" + m
}

func IPID(netid string, a netip.Addr) string {
	// Even public addresses are scoped to the observing network: NAT and
	// anycast make IP↔device bindings local. Relating them globally is phase 8.
	if netid == "" || !a.IsValid() || a.IsUnspecified() || a.IsMulticast() || a.IsLoopback() {
		return ""
	}
	a = a.Unmap().WithZone("")
	return "ip:" + netid + "/" + a.String()
}

// IsIPLike reports address identifiers, whose link to a device is
// time-bounded (spec §6.2): on-link ip, routed rip, global gip.
func IsIPLike(id string) bool {
	switch Kind(id) {
	case "ip", "rip", "gip":
		return true
	}
	return false
}

var nonPublic = []netip.Prefix{
	netip.MustParsePrefix("100.64.0.0/10"),   // CGNAT
	netip.MustParsePrefix("192.0.0.0/24"),    // IETF protocol assignments
	netip.MustParsePrefix("198.18.0.0/15"),   // benchmarking
	netip.MustParsePrefix("192.0.2.0/24"),    // documentation
	netip.MustParsePrefix("198.51.100.0/24"), // documentation
	netip.MustParsePrefix("203.0.113.0/24"),  // documentation
	netip.MustParsePrefix("2001:db8::/32"),   // documentation
}

// IsPublic reports a globally routable unicast address.
func IsPublic(a netip.Addr) bool {
	a = a.Unmap()
	if !a.IsGlobalUnicast() || a.IsPrivate() {
		return false
	}
	for _, p := range nonPublic {
		if p.Contains(a) {
			return false
		}
	}
	return true
}

// RoutedIPID identifies an address that is not on an attached network (a
// router hop, a host in another VLAN). Public addresses are global (gip:)
// so that maps from different networks meet on shared internet paths;
// private ones are only meaningful as seen from the observing network (rip:).
func RoutedIPID(netid string, a netip.Addr) string {
	if !a.IsValid() || a.IsUnspecified() || a.IsMulticast() || a.IsLoopback() {
		return ""
	}
	a = a.Unmap().WithZone("")
	if IsPublic(a) {
		return "gip:" + a.String()
	}
	if netid == "" {
		return ""
	}
	return "rip:" + netid + "/" + a.String()
}

var nameBlock = map[string]bool{
	"": true, "localhost": true, "localhost.localdomain": true, "localhost.local": true,
}

func NameID(kind, netid, name string) string {
	name = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(name)), ".")
	if nameBlock[name] || netid == "" {
		return ""
	}
	return kind + ":" + netid + "/" + name
}

func RKMID(pub ed25519.PublicKey) string { return "rkm:" + hex.EncodeToString(pub) }

func NetEntityID(netid string) string { return "net:" + netid }

// Idents filters empty strings out of a list of identifiers.
func Idents(ids ...string) []string {
	out := ids[:0:0]
	for _, id := range ids {
		if id != "" {
			out = append(out, id)
		}
	}
	return out
}
