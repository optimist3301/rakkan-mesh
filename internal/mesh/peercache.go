package mesh

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"
)

// maxCachedPeers bounds the peer cache; the most recently seen members win.
const maxCachedPeers = 200

// cachedPeer is a member's last known addresses. The cache only says where
// to dial: membership is verified again on every connection, so a stale or
// tampered cache cannot admit anyone.
type cachedPeer struct {
	ID       peer.ID   `json:"id"`
	Addrs    []string  `json:"addrs"`
	LastSeen time.Time `json:"last_seen"`
}

// loadPeerCache adds cached members as dial targets, so a restarted node
// can rejoin even when every bootstrap peer is gone.
func (n *Node) loadPeerCache() {
	if n.cfg.PeerCache == "" {
		return
	}
	b, err := os.ReadFile(n.cfg.PeerCache)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	var peers []cachedPeer
	if err == nil {
		err = json.Unmarshal(b, &peers)
	}
	if err != nil {
		n.log.Warn("ignoring peer cache", "path", n.cfg.PeerCache, "err", err)
		return
	}
	for _, cp := range peers[:min(len(peers), maxCachedPeers)] {
		ai := peer.AddrInfo{ID: cp.ID}
		for _, s := range cp.Addrs {
			if a, err := ma.NewMultiaddr(s); err == nil {
				ai.Addrs = append(ai.Addrs, a)
			}
		}
		n.addDialTarget(ai)
	}
	n.log.Info("loaded peer cache", "peers", len(peers))
}

// savePeerCache writes known members and their addresses atomically.
func (n *Node) savePeerCache() {
	if n.cfg.PeerCache == "" {
		return
	}
	var peers []cachedPeer
	members := n.Members()
	// Members is sorted oldest first; keep the most recent.
	for i := len(members) - 1; i >= 0 && len(peers) < maxCachedPeers; i-- {
		m := members[i]
		// These are the addresses members advertise, which already omit
		// loopback unless that is all they have (see withoutLoopback).
		cp := cachedPeer{ID: m.ID, LastSeen: m.LastSeen}
		for _, a := range n.host.Peerstore().Addrs(m.ID) {
			cp.Addrs = append(cp.Addrs, a.String())
		}
		if len(cp.Addrs) > 0 {
			peers = append(peers, cp)
		}
	}
	b, err := json.MarshalIndent(peers, "", "  ")
	if err != nil {
		return
	}
	tmp, err := os.CreateTemp(filepath.Dir(n.cfg.PeerCache), ".peers-*")
	if err != nil {
		n.log.Warn("saving peer cache", "err", err)
		return
	}
	_, err = tmp.Write(b)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp.Name(), n.cfg.PeerCache)
	}
	if err != nil {
		os.Remove(tmp.Name())
		n.log.Warn("saving peer cache", "err", err)
	}
}
