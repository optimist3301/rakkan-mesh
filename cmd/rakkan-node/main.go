// Command rakkan-node is the Rakkan mesh node: it joins a closed network,
// manages its membership, and executes WebAssembly tasks in the sandbox.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/isolate"
	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// version is set at release build time: -ldflags "-X main.version=...".
var version = "dev"

const usage = `usage: rakkan-node <command> [flags]

commands:
  daemon            join a network and keep running (-config node.json)
  config init       write a JSON config with the node's key and certificate
  id                print this node's peer ID, creating its key if needed
  network init      create a network: its root key and ID
  network issue     issue a node certificate
  network revoke    revoke certificates or peers
  network open      admit guests that prove work on their identity
  network close     stop admitting guests
  network show      print a certificate, revocation list or policy
  network announce  publish in the Mainline DHT the members newcomers find
  run <task.wasm>   execute a task locally in the sandbox
  update            install the latest release from GitHub (-check: only look)
  version           print the version
`

func main() {
	// The node re-executes itself as sandbox worker processes.
	if isolate.IsWorker() {
		if err := isolate.RunWorker(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	switch os.Args[1] {
	case "run":
		os.Exit(runCmd(os.Args[2:]))
	case "id":
		os.Exit(idCmd(os.Args[2:]))
	case "daemon":
		code := daemonCmd(os.Args[2:])
		if restartPending {
			restartSelf()
			code = exitRestart
		}
		os.Exit(code)
	case "update":
		os.Exit(updateCmd(os.Args[2:]))
	case "version", "-version", "--version":
		fmt.Println("rakkan-node", version, runtime.GOOS+"/"+runtime.GOARCH, runtime.Version())
		os.Exit(0)
	case "config":
		os.Exit(configCmd(os.Args[2:]))
	case "network":
		os.Exit(networkCmd(os.Args[2:]))
	default:
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
}

func runCmd(args []string) int {
	def := sandbox.DefaultLimits()
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	entry := fs.String("entry", sandbox.DefaultEntry, "exported entry point")
	inputPath := fs.String("input", "", "input file, or - for stdin")
	memMiB := fs.Uint("mem", uint(def.MaxMemoryPages)*sandbox.PageSize>>20, "memory limit in MiB")
	gas := fs.Int64("gas", def.MaxGas, "gas budget")
	timeout := fs.Duration("timeout", def.Timeout, "wall clock limit")
	isolation := fs.String("isolation", "strict", "worker process isolation: strict, ns, process or none (in-process)")
	cgroup := fs.String("cgroup", "", `delegated cgroup v2 directory for per-worker limits, or "self"`)
	fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "run: expected exactly one module path")
		return 2
	}

	module, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	var input []byte
	switch *inputPath {
	case "":
	case "-":
		input, err = io.ReadAll(os.Stdin)
	default:
		input, err = os.ReadFile(*inputPath)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}

	lim := def
	lim.MaxMemoryPages = uint32(*memMiB << 20 / sandbox.PageSize)
	lim.MaxGas = *gas
	lim.Timeout = *timeout

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	// Local runs are trusted by the operator, so the requested limits are
	// also the policy ceiling.
	task := sandbox.Task{Module: module, Entry: *entry, Input: input, Limits: lim}
	var res *sandbox.Result
	if *isolation == "none" {
		r := sandbox.NewRunner(lim)
		defer r.Close(context.Background())
		res, err = r.Run(ctx, task)
	} else {
		var iso isolate.Isolation
		switch *isolation {
		case "strict":
			iso = isolate.StrictIsolation()
		case "ns":
			iso = isolate.Isolation{Namespaces: true}
		case "process":
		default:
			fmt.Fprintf(os.Stderr, "run: unknown isolation %q\n", *isolation)
			return 2
		}
		iso.Cgroup = *cgroup
		var pool *isolate.Pool
		pool, err = isolate.NewPool(isolate.Config{Ceiling: lim, Workers: 1, Isolation: iso})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer pool.Close()
		res, err = pool.Run(ctx, "", task)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "task failed:", err)
		return 1
	}
	os.Stdout.Write(res.Output)
	if len(res.Logs) > 0 {
		os.Stderr.Write(res.Logs)
		if res.LogTruncated {
			fmt.Fprintln(os.Stderr, "\n[log truncated]")
		}
	}
	fmt.Fprintf(os.Stderr, "exit=%d gas=%d elapsed=%s\n", res.ExitCode, res.GasUsed, res.Elapsed.Round(time.Microsecond))
	if res.ExitCode != 0 {
		return 3
	}
	return 0
}
