//go:build !windows

package isolate

import (
	"os"
	"os/exec"
)

// inheritFiles hands files to the worker as descriptors 3, 4, ...
func inheritFiles(cmd *exec.Cmd, files []*os.File) { cmd.ExtraFiles = files }
