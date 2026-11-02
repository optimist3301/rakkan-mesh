//go:build !linux

package isolate

import (
	"errors"
	"fmt"
	"syscall"
)

// Process isolation is implemented for Linux only. Elsewhere workers still
// run in separate processes, which contains crashes but not escapes.

func (iso Isolation) startStage() int { return 2 }

func (iso Isolation) sysProcAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{} }

func checkPlatform(iso Isolation) error {
	if iso.Namespaces || iso.Seccomp || iso.Cgroup != "" {
		return errors.New("process isolation is only supported on Linux")
	}
	return nil
}

func harden(iso Isolation, stage int) error {
	if stage != 2 {
		return fmt.Errorf("invalid worker stage %d", stage)
	}
	prepareRuntime(iso)
	return nil
}

type cgroupRoot struct{}

type workerCgroup struct{}

func openCgroupRoot(string) (*cgroupRoot, error) { return nil, errors.New("cgroups require Linux") }

func (*cgroupRoot) newWorker(Isolation) (*workerCgroup, error) {
	return nil, errors.New("cgroups require Linux")
}

func (*workerCgroup) oomKilled() bool { return false }

func (*workerCgroup) remove() {}

func setCgroupFD(*syscall.SysProcAttr, *workerCgroup) {}

// isSeccompKill reports whether sig is how seccomp kills a worker; there is
// no seccomp outside Linux.
func isSeccompKill(sig syscall.Signal) bool { return false }
