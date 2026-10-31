package mesh

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	drouting "github.com/libp2p/go-libp2p/p2p/discovery/routing"
	"github.com/libp2p/go-libp2p/p2p/discovery/util"
	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
	ma "github.com/multiformats/go-multiaddr"
)

// maxPingFailures closes a connection after this many consecutive failed
// pings, so the member is redialled on a fresh connection.
const maxPingFailures = 3

// Member describes a peer that has authenticated at least once.
type Member struct {
	ID        peer.ID
	Roles     []string
	Connected bool
	// Relayed is set when every connection to the member goes through a
	// relay; Addr is the remote address of the preferred connection.
	Relayed  bool
	Addr     string
	RTT      time.Duration
	LastSeen time.Time
}

// Members lists known members, connected or not.
func (n *Node) Members() []Member {
	n.mu.Lock()
	var out []Member
	for p, st := range n.peers {
		if st.cert == nil {
			continue
		}
		out = append(out, Member{ID: p, Roles: st.cert.Roles, Connected: st.authed, RTT: st.rtt, LastSeen: st.lastSeen})
	}
	n.mu.Unlock()
	for i := range out {
		conns := n.host.Network().ConnsToPeer(out[i].ID)
		if len(conns) == 0 {
			continue
		}
		out[i].Relayed = true
		out[i].Addr = conns[0].RemoteMultiaddr().String()
		for _, c := range conns {
			if !isRelayed(c) {
				out[i].Relayed, out[i].Addr = false, c.RemoteMultiaddr().String()
				break
			}
		}
	}
	slices.SortFunc(out, func(a, b Member) int { return a.LastSeen.Compare(b.LastSeen) })
	return out
}

func isRelayed(c network.Conn) bool {
	_, err := c.RemoteMultiaddr().ValueForProtocol(ma.P_CIRCUIT)
	return err == nil
}

// healthLoop pings connected members, closes connections that stop
// responding, and drops members whose certificates no longer verify.
func (n *Node) healthLoop() {
	tick := time.NewTicker(n.cfg.HealthInterval)
	defer tick.Stop()
	lastSave := time.Now()
	for {
		select {
		case <-n.ctx.Done():
			return
		case <-tick.C:
		}
		// Save periodically too: a crash skips the save in Close.
		if time.Since(lastSave) > 30*time.Second {
			n.savePeerCache()
			lastSave = time.Now()
		}
		for _, m := range n.Members() {
			if m.Connected {
				go n.checkMember(m.ID)
			}
		}
	}
}

func (n *Node) checkMember(p peer.ID) {
	n.mu.Lock()
	signed := n.peers[p].signed
	n.mu.Unlock()
	if _, err := n.cfg.Network.Verify(signed, time.Now(), n.revs.Revoked); err != nil {
		n.deny(p, denyAfterFailure, err) // expired or revoked since admission
		return
	}

	ctx, cancel := context.WithTimeout(n.ctx, n.cfg.HealthInterval)
	defer cancel()
	res := <-ping.Ping(ctx, n.host, p)
	n.mu.Lock()
	st := n.peers[p]
	if res.Error == nil {
		st.rtt, st.lastSeen, st.failures = res.RTT, time.Now(), 0
		n.mu.Unlock()
		return
	}
	st.failures++
	dead := st.failures >= maxPingFailures
	n.mu.Unlock()
	if dead {
		n.log.Warn("member unresponsive, reconnecting", "peer", p.ShortString(), "err", res.Error)
		n.host.Network().ClosePeer(p)
	}
}

// addDialTarget remembers a peer to connect to and keep connected.
func (n *Node) addDialTarget(ai peer.AddrInfo) {
	if ai.ID == n.host.ID() {
		return
	}
	n.host.Peerstore().AddAddrs(ai.ID, ai.Addrs, peerstore.PermanentAddrTTL)
	n.mu.Lock()
	st := n.state(ai.ID)
	if !st.dial {
		st.dial, st.nextDial = true, time.Now()
	}
	n.mu.Unlock()
}

// dialLoop reconnects to dial targets with exponential backoff.
func (n *Node) dialLoop() {
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-n.ctx.Done():
			return
		case <-tick.C:
		}
		now := time.Now()
		var due []peer.ID
		n.mu.Lock()
		for p, st := range n.peers {
			if st.dial && !st.authed && !now.Before(st.nextDial) && !n.authing[p] {
				// Schedule the next attempt before trying, so a slow dial
				// is not started twice.
				st.backoff = min(max(2*st.backoff, time.Second), n.cfg.MaxBackoff)
				jitter := time.Duration(rand.Int64N(int64(st.backoff/4) + 1))
				st.nextDial = now.Add(st.backoff + jitter)
				due = append(due, p)
			}
		}
		n.mu.Unlock()
		for _, p := range due {
			if !n.isDenied(p) {
				go n.redial(p)
			}
		}
	}
}

func (n *Node) redial(p peer.ID) {
	ctx, cancel := context.WithTimeout(n.ctx, 10*time.Second)
	defer cancel()
	if n.host.Network().Connectedness(p) == network.Connected {
		return // authentication is in progress or will be retried
	}
	addrs := n.host.Peerstore().Addrs(p)
	if len(addrs) == 0 && n.dht != nil {
		// The member may have moved; ask the DHT for fresh addresses.
		if ai, err := n.dht.FindPeer(ctx, p); err == nil {
			addrs = ai.Addrs
		}
	}
	if len(addrs) == 0 {
		return
	}
	if err := n.host.Connect(ctx, peer.AddrInfo{ID: p, Addrs: addrs}); err != nil && !errors.Is(err, context.Canceled) {
		n.log.Debug("redial failed", "peer", p.ShortString(), "err", err)
	}
}

// discoveryLoop advertises membership on the DHT and connects to members
// found there. Rounds start every couple of seconds and back off to
// DiscoveryInterval; admitting a new member triggers one at once, since the
// routing table just grew. (util.Advertise is not used: when the first
// attempt fails on an empty routing table it waits two minutes.)
func (n *Node) discoveryLoop() {
	ns := "rakkan/" + n.tag() + "/members"
	disc := drouting.NewRoutingDiscovery(n.dht)
	// Bootstrap populates the routing table; it is retried periodically.
	n.dht.Bootstrap(n.ctx)
	interval := min(2*time.Second, n.cfg.DiscoveryInterval)
	timer := time.NewTimer(interval)
	defer timer.Stop()
	var readvertise time.Time
	for {
		ctx, cancel := context.WithTimeout(n.ctx, n.cfg.DiscoveryInterval)
		if time.Now().After(readvertise) {
			ttl, err := disc.Advertise(ctx, ns)
			n.log.Debug("advertised membership", "err", err)
			if err == nil {
				readvertise = time.Now().Add(7 * ttl / 8)
			}
		}
		peers, err := util.FindPeers(ctx, disc, ns)
		n.log.Debug("discovery round", "found", len(peers), "err", err)
		cancel()
		if err == nil {
			for _, ai := range peers {
				if ai.ID != n.host.ID() && len(ai.Addrs) > 0 {
					n.addDialTarget(ai)
				}
			}
		}
		select {
		case <-n.ctx.Done():
			return
		case <-timer.C:
			interval = min(2*interval, n.cfg.DiscoveryInterval)
		case <-n.discover:
			if !timer.Stop() {
				<-timer.C
			}
		}
		timer.Reset(interval)
	}
}
