//go:build !(cgo && ((linux && (amd64 || arm64)) || (windows && amd64)))

package sandbox

import "errors"

const (
	rxlibName      = "none"
	rxlibAvailable = false
)

var errNoRandomX = errors.New("this build has no RandomX (it needs cgo on linux/amd64, linux/arm64 or windows/amd64, and librandomx: see `make randomx`)")

type rxlibCache struct{}

func newRxlibCache() (*rxlibCache, error) { return nil, errNoRandomX }
func (*rxlibCache) init([]byte)           {}
func (*rxlibCache) close()                {}

type rxlibVM struct{}

func newRxlibVM(*rxlibDataset) (*rxlibVM, error) { return nil, errNoRandomX }
func (*rxlibVM) hash([]byte, *[rxHashSize]byte)  {}
func (*rxlibVM) close()                          {}

type rxlibDataset struct{}

func newRxlibDataset([]byte) (*rxlibDataset, error) { return nil, errNoRandomX }
func (*rxlibDataset) fill(*rxlibCache, int)         {}
