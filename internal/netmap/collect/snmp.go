//go:build linux

package collect

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"strconv"
	"strings"
	"time"

	"github.com/gosnmp/gosnmp"

	"github.com/optimist3301/rakkan-mesh/internal/netmap/obs"
)

// SNMPAuth holds credentials for devices the operator manages. There is no
// guessing: without credentials the collector does nothing.
type SNMPAuth struct {
	Community string // v2c
	V3User    string // v3 (authPriv with SHA/AES when passphrases are given)
	V3Auth    string
	V3Priv    string
}

func (a SNMPAuth) Valid() bool { return a.Community != "" || a.V3User != "" }

// SNMP polls explicitly listed switches/routers for their identity,
// interfaces, forwarding database (which MAC sits behind which port), LLDP
// neighbor table and ARP table. This is what turns a flat device list into
// wired topology.
type SNMP struct {
	Targets []netip.Addr
	Auth    SNMPAuth
	EdgeMax int // ports with more learned MACs are treated as uplinks
	Timeout time.Duration
}

func (SNMP) Name() string { return "snmp" }

// walker abstracts an SNMP walk so the parsing logic is testable offline.
type walker interface {
	walk(oid string) ([]gosnmp.SnmpPDU, error)
}

type gsWalker struct{ g *gosnmp.GoSNMP }

func (w gsWalker) walk(oid string) ([]gosnmp.SnmpPDU, error) {
	if w.g.Version == gosnmp.Version1 {
		return w.g.WalkAll(oid)
	}
	return w.g.BulkWalkAll(oid)
}

func (c SNMP) client(target netip.Addr) *gosnmp.GoSNMP {
	g := &gosnmp.GoSNMP{
		Target: target.String(), Port: 161, Timeout: c.Timeout, Retries: 1,
		MaxRepetitions: 25, Version: gosnmp.Version2c, Community: c.Auth.Community,
	}
	if c.Auth.V3User != "" {
		usm := &gosnmp.UsmSecurityParameters{UserName: c.Auth.V3User}
		g.Version, g.SecurityModel, g.MsgFlags = gosnmp.Version3, gosnmp.UserSecurityModel, gosnmp.NoAuthNoPriv
		if c.Auth.V3Auth != "" {
			usm.AuthenticationProtocol, usm.AuthenticationPassphrase = gosnmp.SHA, c.Auth.V3Auth
			g.MsgFlags = gosnmp.AuthNoPriv
			if c.Auth.V3Priv != "" {
				usm.PrivacyProtocol, usm.PrivacyPassphrase = gosnmp.AES, c.Auth.V3Priv
				g.MsgFlags = gosnmp.AuthPriv
			}
		}
		g.SecurityParameters = usm
	}
	return g
}

func (c SNMP) Run(ctx context.Context, s *Session, nets []NetCtx) error {
	if len(c.Targets) == 0 {
		return nil
	}
	if !c.Auth.Valid() {
		return errors.New("snmp: no credentials given (--snmp-community or --snmp-v3-user)")
	}
	var errs []error
	for _, t := range c.Targets {
		if ctx.Err() != nil {
			break
		}
		n := netFor(nets, t)
		g := c.client(t)
		if err := g.Connect(); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", t, err))
			continue
		}
		st, err := pollDevice(s, n, t, gsWalker{g}, c.EdgeMax)
		g.Conn.Close()
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", t, err))
			continue
		}
		s.Log.Printf("snmp %s: %q, %d interfaces, %d FDB entries, %d LLDP neighbors, %d ARP entries",
			t, st.name, st.ifaces, st.fdb, st.lldp, st.arp)
	}
	return errors.Join(errs...)
}

// netFor picks the attached network a target belongs to (or the first one,
// as the vantage for routed management addresses).
func netFor(nets []NetCtx, a netip.Addr) *NetCtx {
	for i := range nets {
		if nets[i].Contains(a) {
			return &nets[i]
		}
	}
	return &nets[0]
}

// ---- OIDs ----

const (
	oidSystem        = ".1.3.6.1.2.1.1"
	oidIfDescr       = ".1.3.6.1.2.1.2.2.1.2"
	oidIfPhysAddress = ".1.3.6.1.2.1.2.2.1.6"
	oidIfName        = ".1.3.6.1.2.1.31.1.1.1.1"
	oidBasePortIfIdx = ".1.3.6.1.2.1.17.1.4.1.2"
	oidDot1dFdbPort  = ".1.3.6.1.2.1.17.4.3.1.2"
	oidDot1dFdbState = ".1.3.6.1.2.1.17.4.3.1.3"
	oidDot1qFdbPort  = ".1.3.6.1.2.1.17.7.1.2.2.1.2"
	oidDot1qFdbState = ".1.3.6.1.2.1.17.7.1.2.2.1.3"
	oidIPNetToMedia  = ".1.3.6.1.2.1.4.22.1.2"
	oidLLDPLocal     = ".1.0.8802.1.1.2.1.3"
	oidLLDPRem       = ".1.0.8802.1.1.2.1.4.1.1"
)

const fdbStatusSelf = 4 // dot1dTpFdbStatus/dot1qTpFdbStatus "self": the switch's own MAC

func suffix(name, prefix string) (string, bool) {
	rest, ok := strings.CutPrefix(name, prefix+".")
	return rest, ok
}

func toInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case uint:
		return int(x)
	case uint32:
		return int(x)
	case uint64:
		return int(x)
	case int64:
		return int(x)
	}
	return -1
}

func toBytes(v any) []byte {
	if b, ok := v.([]byte); ok {
		return b
	}
	if s, ok := v.(string); ok {
		return []byte(s)
	}
	return nil
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return strings.TrimRight(string(toBytes(v)), "\x00")
}

// macFromOID parses six decimal sub-identifiers into a MAC.
func macFromOID(parts []string) net.HardwareAddr {
	if len(parts) != 6 {
		return nil
	}
	mac := make(net.HardwareAddr, 6)
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 || n > 255 {
			return nil
		}
		mac[i] = byte(n)
	}
	return mac
}

// ChassisID builds the lldp: identifier exactly as the passive LLDP sniffer
// does, so a switch polled over SNMP and one heard on the wire merge.
func ChassisID(subtype int, id []byte) (string, net.HardwareAddr) {
	switch subtype {
	case 4: // macAddress
		if len(id) == 6 {
			mac := net.HardwareAddr(id)
			return "lldp:mac/" + strings.ToLower(mac.String()), mac
		}
	case 5: // networkAddress
		return fmt.Sprintf("lldp:net/%x", id), nil
	}
	if len(id) == 0 {
		return "", nil
	}
	return fmt.Sprintf("lldp:%d/%s", subtype, strings.ToLower(string(id))), nil
}

func portIDString(subtype int, id []byte) string {
	if subtype == 3 && len(id) == 6 { // macAddress
		return net.HardwareAddr(id).String()
	}
	return strings.TrimRight(string(id), "\x00")
}

type pollStats struct {
	name                   string
	ifaces, fdb, lldp, arp int
}

// pollDevice turns one device's MIB tables into observations.
func pollDevice(s *Session, n *NetCtx, target netip.Addr, w walker, edgeMax int) (pollStats, error) {
	var st pollStats
	onLink := n.Contains(target)
	scope := ""
	if onLink {
		scope = n.NetID // random MACs can only be scoped if we share the L2 network
	}
	emit := func(subject []string, claims ...obs.Claim) {
		s.Emit(obs.Body{Mode: obs.ModeActive, Method: "snmp", Vantage: n.Vantage(), Subject: subject, Claims: claims})
	}

	sys, err := w.walk(oidSystem)
	if err != nil {
		return st, err
	}
	if len(sys) == 0 {
		return st, errors.New("no response to system group (wrong credentials?)")
	}
	var descr, objID, location string
	for _, p := range sys {
		switch p.Name {
		case oidSystem + ".1.0":
			descr = toString(p.Value)
		case oidSystem + ".2.0":
			objID = toString(p.Value)
		case oidSystem + ".5.0":
			st.name = toString(p.Value)
		case oidSystem + ".6.0":
			location = toString(p.Value)
		}
	}

	// Interfaces.
	ifMAC := map[int]net.HardwareAddr{}
	ifName := map[int]string{}
	own := map[string]bool{}
	if pdus, err := w.walk(oidIfPhysAddress); err == nil {
		for _, p := range pdus {
			if idx, ok := suffix(p.Name, oidIfPhysAddress); ok {
				if mac := net.HardwareAddr(toBytes(p.Value)); len(mac) == 6 {
					i, _ := strconv.Atoi(idx)
					ifMAC[i] = mac
					own[mac.String()] = true
				}
			}
		}
	}
	for _, oid := range []string{oidIfDescr, oidIfName} { // ifName wins over ifDescr
		if pdus, err := w.walk(oid); err == nil {
			for _, p := range pdus {
				if idx, ok := suffix(p.Name, oid); ok {
					i, _ := strconv.Atoi(idx)
					if v := toString(p.Value); v != "" {
						ifName[i] = v
					}
				}
			}
		}
	}
	st.ifaces = len(ifName)

	// LLDP local chassis and port names.
	var chassisSub int
	var chassisRaw []byte
	locPort := map[int]string{}
	if pdus, err := w.walk(oidLLDPLocal); err == nil {
		for _, p := range pdus {
			rest, _ := suffix(p.Name, oidLLDPLocal)
			switch {
			case rest == "1.0":
				chassisSub = toInt(p.Value)
			case rest == "2.0":
				chassisRaw = toBytes(p.Value)
			case strings.HasPrefix(rest, "7.1.4."): // lldpLocPortDesc
				i, _ := strconv.Atoi(strings.TrimPrefix(rest, "7.1.4."))
				if v := toString(p.Value); v != "" {
					locPort[i] = v
				}
			case strings.HasPrefix(rest, "7.1.3."): // lldpLocPortId
				i, _ := strconv.Atoi(strings.TrimPrefix(rest, "7.1.3."))
				if _, ok := locPort[i]; !ok {
					locPort[i] = toString(p.Value)
				}
			}
		}
	}

	// Device identity.
	chassis, chassisMAC := ChassisID(chassisSub, chassisRaw)
	ids := []string{chassis, obs.MACID(chassisMAC, scope), n.AddrID(target)}
	seenMAC := map[string]bool{}
	for _, mac := range ifMAC {
		if id := obs.MACID(mac, scope); id != "" && !seenMAC[id] {
			seenMAC[id] = true
			ids = append(ids, id)
		}
	}
	ids = obs.Idents(ids...)
	self := ""
	for _, id := range ids { // anchor for links: strongest identifier
		if obs.StrengthOf(id) == obs.Strong {
			self = id
			break
		}
	}
	if self == "" {
		self = n.AddrID(target)
	}

	// Bridge: port → ifIndex, then the forwarding database.
	portIf := map[int]int{}
	if pdus, err := w.walk(oidBasePortIfIdx); err == nil {
		for _, p := range pdus {
			if idx, ok := suffix(p.Name, oidBasePortIfIdx); ok {
				bp, _ := strconv.Atoi(idx)
				portIf[bp] = toInt(p.Value)
			}
		}
	}
	type fdbEntry struct {
		mac  net.HardwareAddr
		port int
	}
	var fdb []fdbEntry
	readFDB := func(portOID, stateOID string, macAt int) {
		state := map[string]int{}
		if pdus, err := w.walk(stateOID); err == nil {
			for _, p := range pdus {
				if idx, ok := suffix(p.Name, stateOID); ok {
					state[idx] = toInt(p.Value)
				}
			}
		}
		pdus, err := w.walk(portOID)
		if err != nil {
			return
		}
		for _, p := range pdus {
			idx, ok := suffix(p.Name, portOID)
			if !ok || state[idx] == fdbStatusSelf {
				continue
			}
			parts := strings.Split(idx, ".")
			if len(parts) < macAt+6 {
				continue
			}
			if mac := macFromOID(parts[macAt : macAt+6]); mac != nil && !own[mac.String()] {
				fdb = append(fdb, fdbEntry{mac, toInt(p.Value)})
			}
		}
	}
	readFDB(oidDot1qFdbPort, oidDot1qFdbState, 1) // index: fdbId.mac
	if len(fdb) == 0 {
		readFDB(oidDot1dFdbPort, oidDot1dFdbState, 0) // index: mac
	}
	st.fdb = len(fdb)

	// LLDP neighbors: index timeMark.localPort.remIndex, columns .4–.10.
	type rem struct {
		localPort                  int
		chSub, portSub             int
		chassis, port              []byte
		portDesc, sysName, sysDesc string
	}
	rems := map[string]*rem{}
	if pdus, err := w.walk(oidLLDPRem); err == nil {
		for _, p := range pdus {
			rest, _ := suffix(p.Name, oidLLDPRem)
			parts := strings.Split(rest, ".")
			if len(parts) != 4 {
				continue
			}
			key := parts[2] + "." + parts[3]
			r := rems[key]
			if r == nil {
				lp, _ := strconv.Atoi(parts[2])
				r = &rem{localPort: lp}
				rems[key] = r
			}
			switch parts[0] {
			case "4":
				r.chSub = toInt(p.Value)
			case "5":
				r.chassis = toBytes(p.Value)
			case "6":
				r.portSub = toInt(p.Value)
			case "7":
				r.port = toBytes(p.Value)
			case "8":
				r.portDesc = toString(p.Value)
			case "9":
				r.sysName = toString(p.Value)
			case "10":
				r.sysDesc = toString(p.Value)
			}
		}
	}
	st.lldp = len(rems)

	// A port with an LLDP neighbor leads to infrastructure, not an end host.
	// lldpLocPortNum is vendor-specific: it may equal the bridge port or the
	// ifIndex, so both readings mark the port as an uplink.
	uplinkIf := map[int]bool{}
	for _, r := range rems {
		uplinkIf[r.localPort] = true
		if ifi, ok := portIf[r.localPort]; ok {
			uplinkIf[ifi] = true
		}
	}

	claims := []obs.Claim{claim("name", st.name), claim("os.desc", descr), claim("snmp.oid", objID),
		claimC("location.text", location, 600)}
	if len(portIf) > 0 {
		claims = append(claims, claim("role", "switch"))
	}
	emit(ids, claims...)

	// FDB → which host is behind which port.
	perPort := map[int]int{}
	for _, e := range fdb {
		perPort[e.port]++
	}
	for _, e := range fdb {
		id := obs.MACID(e.mac, scope)
		if id == "" {
			continue
		}
		ifi, ok := portIf[e.port]
		if !ok {
			ifi = e.port
		}
		pname := ifName[ifi]
		if pname == "" {
			pname = "port " + strconv.Itoa(e.port)
		}
		if uplinkIf[ifi] || perPort[e.port] > edgeMax {
			// Many MACs (or an LLDP neighbor) on this port: the host is
			// somewhere behind it, not plugged into it.
			emit([]string{id}, claimC("link.l2-behind", self, 300), claim("link.port.b", pname), claim("seen.l2", "snmp-fdb"))
		} else {
			emit([]string{id}, claimC("link.l2", self, 800), claim("link.port.b", pname), claim("seen.l2", "snmp-fdb"))
		}
	}

	// LLDP neighbors → direct links with port names on both sides.
	for _, r := range rems {
		rid, rmac := ChassisID(r.chSub, r.chassis)
		local := locPort[r.localPort]
		if local == "" {
			local = ifName[r.localPort]
		}
		emit(obs.Idents(rid, obs.MACID(rmac, scope)),
			claim("name", r.sysName), claim("os.desc", r.sysDesc),
			claim("link.l2", self), claim("link.port.a", portIDString(r.portSub, r.port)), claim("link.port.b", local),
			claim("lldp.port.desc", r.portDesc))
	}

	// ARP table (routers): MAC ↔ IP, including subnets we are not attached to.
	if pdus, err := w.walk(oidIPNetToMedia); err == nil {
		for _, p := range pdus {
			idx, ok := suffix(p.Name, oidIPNetToMedia)
			if !ok {
				continue
			}
			parts := strings.SplitN(idx, ".", 2)
			if len(parts) != 2 {
				continue
			}
			ip, err := netip.ParseAddr(parts[1])
			mac := net.HardwareAddr(toBytes(p.Value))
			if err != nil || len(mac) != 6 || own[mac.String()] {
				continue
			}
			st.arp++
			emit(obs.Idents(obs.MACID(mac, scope), n.AddrID(ip)), claim("seen.l3", "snmp-arp"))
		}
	}
	return st, nil
}
