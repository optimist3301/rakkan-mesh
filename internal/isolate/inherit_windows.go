package isolate

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	"golang.org/x/sys/windows"
)

// inheritFiles hands the files' handles to the worker: Windows has no
// numbered descriptors and exec rejects ExtraFiles, so the handles are made
// inheritable, listed for the child, and named in an environment variable
// (decimal, comma separated, in order) that sandbox.InheritedHandle reads.
func inheritFiles(cmd *exec.Cmd, files []*os.File) {
	var hs []syscall.Handle
	var names []string
	for _, f := range files {
		h := windows.Handle(f.Fd())
		if windows.SetHandleInformation(h, windows.HANDLE_FLAG_INHERIT, windows.HANDLE_FLAG_INHERIT) != nil {
			continue
		}
		hs = append(hs, syscall.Handle(h))
		names = append(names, strconv.FormatUint(uint64(h), 10))
	}
	if len(hs) == 0 {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.AdditionalInheritedHandles = hs
	cmd.Env = append(cmd.Env, sandbox.InheritedHandlesEnv+"="+strings.Join(names, ","))
}
