package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/update"
)

// exitRestart is the exit code of a daemon that updated itself but could not
// restart in place (Windows): a service manager that restarts failed
// programs starts the new version.
const exitRestart = 75

// restartPending is set by a daemon that has installed an update and wants
// to be started again, now that it has shut down.
var restartPending bool

// minUpdateInterval keeps nodes from hammering the GitHub API (60 requests
// an hour per address without a token).
const minUpdateInterval = time.Hour

// selfPath is the file of the running program, with links followed.
func selfPath() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(exe)
}

// autoUpdate checks for a newer release now and then and installs it,
// asking for a restart through restart. It returns when ctx ends.
func autoUpdate(ctx context.Context, u *update.Updater, interval time.Duration, log *slog.Logger, restart func()) {
	exe, err := selfPath()
	if err != nil {
		log.Warn("auto-update off", "err", err)
		return
	}
	// Start a little after launch, and spread the checks of nodes that
	// were started together.
	wait := time.Minute + rand.N(5*time.Minute)
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
		wait = interval + rand.N(interval/10+1)
		rel, err := u.Latest(ctx, version)
		if err != nil {
			log.Warn("update check failed", "err", err)
			continue
		}
		if rel == nil {
			log.Debug("up to date", "version", version)
			continue
		}
		log.Info("installing update", "from", version, "to", rel.Version)
		if err := u.Apply(ctx, rel, exe); err != nil {
			log.Warn("update failed", "to", rel.Version, "err", err)
			continue
		}
		log.Info("updated; restarting", "to", rel.Version)
		restart()
		return
	}
}

// updateCmd implements "rakkan-node update": check for a release and install it.
func updateCmd(args []string) int {
	fs := flag.NewFlagSet("update", flag.ExitOnError)
	repo := fs.String("repo", update.DefaultRepo, "GitHub repository (owner/name) to take releases from")
	check := fs.Bool("check", false, "only say whether a newer release exists")
	fs.Parse(args)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	u := &update.Updater{Repo: *repo}
	rel, err := u.Latest(ctx, version)
	if err != nil {
		return fail(err)
	}
	if rel == nil {
		fmt.Println("rakkan-node", version, "is up to date")
		return 0
	}
	if *check {
		fmt.Println("newer release available:", rel.Version, "(installed:", version+")")
		return 0
	}
	exe, err := selfPath()
	if err != nil {
		return fail(err)
	}
	fmt.Println("installing", rel.Version, "over", exe)
	if err := u.Apply(ctx, rel, exe); err != nil {
		return fail(err)
	}
	fmt.Println("updated to", rel.Version+"; restart a running node to use it")
	return 0
}
