package isolate

import "testing"

func newRandomXPool(t *testing.T, cfg Config) *Pool {
	cfg.Isolation = StrictIsolation()
	return newPool(t, cfg)
}
