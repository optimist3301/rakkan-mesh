//go:build !windows

package main

import (
	"os"
	"syscall"
)

// restartSelf replaces this process with the program's new version, under
// the same PID, so a service manager notices nothing. It returns only if
// that fails.
func restartSelf() {
	exe, err := selfPath()
	if err != nil {
		return
	}
	syscall.Exec(exe, os.Args, os.Environ())
}
