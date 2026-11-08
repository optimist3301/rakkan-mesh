//go:build windows

package main

// restartSelf cannot replace a process on Windows; the daemon exits with
// exitRestart and the scheduled task, which restarts failures, starts the
// new version.
func restartSelf() {}
