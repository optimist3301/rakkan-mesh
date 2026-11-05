package ygg

import (
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"net"
	"time"

	iwt "github.com/Arceliar/ironwood/types"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/transport"
	ma "github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"
	"github.com/quic-go/quic-go"
)

// Transport is a libp2p transport for /ygg/<key> addresses. Each libp2p
// connection is one QUIC connection over Yggdrasil carrying one stream,
// which libp2p's upgrader secures and multiplexes like a TCP connection.
type Transport struct {
	n     *Node
	up    transport.Upgrader
	rcmgr network.ResourceManager
}

var _ transport.Transport = (*Transport)(nil)

// NewTransport returns a constructor for libp2p.Transport:
//
//	libp2p.Transport(ygg.NewTransport(node))
func NewTransport(n *Node) func(transport.Upgrader, network.ResourceManager) (*Transport, error) {
	return func(up transport.Upgrader, rcmgr network.ResourceManager) (*Transport, error) {
		if rcmgr == nil {
			rcmgr = &network.NullResourceManager{}
		}
		return &Transport{n: n, up: up, rcmgr: rcmgr}, nil
	}
}

func (t *Transport) CanDial(a ma.Multiaddr) bool {
	_, ok := KeyOf(a)
	return ok
}

func (t *Transport) Protocols() []int { return []int{P_YGG} }
func (t *Transport) Proxy() bool      { return false }

func (t *Transport) Dial(ctx context.Context, raddr ma.Multiaddr, p peer.ID) (transport.CapableConn, error) {
	key, ok := KeyOf(raddr)
	if !ok {
		return nil, fmt.Errorf("ygg: cannot dial %s", raddr)
	}
	scope, err := t.rcmgr.OpenConnection(network.DirOutbound, false, raddr)
	if err != nil {
		return nil, err
	}
	if err := scope.SetPeer(p); err != nil {
		scope.Done()
		return nil, err
	}
	c, err := t.dialConn(ctx, key)
	if err != nil {
		scope.Done()
		return nil, err
	}
	cc, err := t.up.Upgrade(ctx, t, c, network.DirOutbound, p, scope)
	if err != nil {
		c.Close()
		return nil, err
	}
	return cc, nil
}

func (t *Transport) dialConn(ctx context.Context, key ed25519.PublicKey) (*conn, error) {
	qc, err := t.n.dial(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("ygg: %w", err)
	}
	st, err := qc.OpenStreamSync(ctx)
	if err != nil {
		qc.CloseWithError(0, "")
		return nil, fmt.Errorf("ygg: %w", err)
	}
	return newConn(t.n.key, qc, st), nil
}

func (t *Transport) Listen(laddr ma.Multiaddr) (transport.Listener, error) {
	key, ok := KeyOf(laddr)
	if !ok {
		return nil, fmt.Errorf("ygg: cannot listen on %s", laddr)
	}
	if !key.Equal(t.n.key) {
		return nil, fmt.Errorf("ygg: %s is not this node's address %s", laddr, t.n.Multiaddr())
	}
	ql, err := t.n.listen()
	if err != nil {
		return nil, fmt.Errorf("ygg: %w", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	l := &listener{n: t.n, ql: ql, ctx: ctx, cancel: cancel, conns: make(chan *conn)}
	go l.run()
	return t.up.UpgradeGatedMaListener(t, t.up.GateMaListener(l)), nil
}

// listener accepts QUIC connections and their one stream. Waiting for the
// stream runs per connection, so a peer that never opens one holds up
// nobody else.
type listener struct {
	n      *Node
	ql     *quic.Listener
	ctx    context.Context
	cancel context.CancelFunc
	conns  chan *conn
}

const streamTimeout = 20 * time.Second

func (l *listener) run() {
	for {
		qc, err := l.ql.Accept(l.ctx)
		if err != nil {
			l.cancel()
			return
		}
		go func() {
			ctx, cancel := context.WithTimeout(l.ctx, streamTimeout)
			defer cancel()
			st, err := qc.AcceptStream(ctx)
			if err != nil {
				qc.CloseWithError(0, "")
				return
			}
			c := newConn(l.n.key, qc, st)
			select {
			case l.conns <- c:
			case <-l.ctx.Done():
				c.Close()
			}
		}()
	}
}

func (l *listener) Accept() (manet.Conn, error) {
	select {
	case c := <-l.conns:
		return c, nil
	case <-l.ctx.Done():
		return nil, net.ErrClosed
	}
}

func (l *listener) Close() error {
	l.cancel()
	l.n.unlisten(l.ql)
	return nil
}

func (l *listener) Addr() net.Addr          { return iwt.Addr(l.n.key) }
func (l *listener) Multiaddr() ma.Multiaddr { return l.n.Multiaddr() }

// conn is a QUIC stream seen as a manet.Conn.
type conn struct {
	*quic.Stream
	qc            *quic.Conn
	local, remote ed25519.PublicKey
}

func newConn(self ed25519.PublicKey, qc *quic.Conn, st *quic.Stream) *conn {
	remote, _ := qc.RemoteAddr().(iwt.Addr)
	return &conn{Stream: st, qc: qc, local: self, remote: ed25519.PublicKey(remote)}
}

// Close ends the stream, letting written data reach the peer, then closes
// the QUIC connection once the peer has closed it too or after a while.
func (c *conn) Close() error {
	err := c.Stream.Close()
	c.Stream.CancelRead(0)
	go func() {
		select {
		case <-c.qc.Context().Done():
		case <-time.After(5 * time.Second):
		}
		c.qc.CloseWithError(0, "")
	}()
	if errors.Is(err, net.ErrClosed) {
		return nil
	}
	return err
}

func (c *conn) LocalAddr() net.Addr               { return iwt.Addr(c.local) }
func (c *conn) RemoteAddr() net.Addr              { return iwt.Addr(c.remote) }
func (c *conn) LocalMultiaddr() ma.Multiaddr      { return Multiaddr(c.local) }
func (c *conn) RemoteMultiaddr() ma.Multiaddr     { return Multiaddr(c.remote) }
func (c *conn) SetDeadline(t time.Time) error     { return c.Stream.SetDeadline(t) }
func (c *conn) SetReadDeadline(t time.Time) error { return c.Stream.SetReadDeadline(t) }
func (c *conn) SetWriteDeadline(t time.Time) error {
	return c.Stream.SetWriteDeadline(t)
}
