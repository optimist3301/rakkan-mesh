package ygg

import (
	"crypto/ed25519"
	"encoding/hex"
	"fmt"

	ma "github.com/multiformats/go-multiaddr"
)

// P_YGG is the multiaddr protocol /ygg/<key>: a node reached over
// Yggdrasil by its 32-byte Yggdrasil key, written in hex. The code is in the
// multicodec private-use range; the protocol is ours, not a standard one.
const P_YGG = 0x3f7967

func init() {
	err := ma.AddProtocol(ma.Protocol{
		Name:  "ygg",
		Code:  P_YGG,
		VCode: ma.CodeToVarint(P_YGG),
		Size:  8 * ed25519.PublicKeySize,
		Transcoder: ma.NewTranscoderFromFunctions(
			func(s string) ([]byte, error) {
				b, err := hex.DecodeString(s)
				if err != nil {
					return nil, fmt.Errorf("ygg: key %q: %w", s, err)
				}
				if len(b) != ed25519.PublicKeySize {
					return nil, fmt.Errorf("ygg: key %q is not %d bytes", s, ed25519.PublicKeySize)
				}
				return b, nil
			},
			func(b []byte) (string, error) { return hex.EncodeToString(b), nil },
			func(b []byte) error {
				if len(b) != ed25519.PublicKeySize {
					return fmt.Errorf("ygg: key is %d bytes", len(b))
				}
				return nil
			}),
	})
	if err != nil {
		panic(err)
	}
}

// Multiaddr is /ygg/<key>.
func Multiaddr(key ed25519.PublicKey) ma.Multiaddr {
	a, err := ma.NewComponent("ygg", hex.EncodeToString(key))
	if err != nil {
		panic(err)
	}
	return a.Multiaddr()
}

// KeyOf returns the key of an address that is exactly /ygg/<key>.
func KeyOf(a ma.Multiaddr) (ed25519.PublicKey, bool) {
	if len(a) != 1 || a[0].Code() != P_YGG {
		return nil, false
	}
	return ed25519.PublicKey(a[0].RawValue()), true
}
