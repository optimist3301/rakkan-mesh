package sandbox

import (
	"os"

	"golang.org/x/sys/unix"
)

func newSharedFile(size int64, huge bool) (*os.File, error) {
	flags := unix.MFD_CLOEXEC
	if huge {
		flags |= unix.MFD_HUGETLB | unix.MFD_HUGE_2MB
	}
	fd, err := unix.MemfdCreate("rakkan-randomx-dataset", flags)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), "rakkan-randomx-dataset")
	if err := f.Truncate(size); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

func mapShared(fd int, off, length int64, write bool) ([]byte, error) {
	prot := unix.PROT_READ
	if write {
		prot |= unix.PROT_WRITE
	}
	return unix.Mmap(fd, off, int(length), prot, unix.MAP_SHARED)
}

func fdSize(fd int) (int64, bool) {
	var st unix.Stat_t
	if unix.Fstat(fd, &st) != nil || st.Mode&unix.S_IFMT != unix.S_IFREG {
		return 0, false
	}
	return st.Size, true
}

// sharedFD is where a worker finds the dataset.
const sharedFD = 3

// inheritedShared is the descriptor of the dataset the node passed, if any.
func inheritedShared() (int, bool) {
	size, ok := fdSize(sharedFD)
	return sharedFD, ok && size == RandomXSharedBytes
}

func unmapShared(b []byte) { unix.Munmap(b) }

// releasePages gives the pages of the file back to the system; the file keeps
// its size and reads as zeros.
func releasePages(f *os.File, off, length int64) error {
	return unix.Fallocate(int(f.Fd()), unix.FALLOC_FL_PUNCH_HOLE|unix.FALLOC_FL_KEEP_SIZE, off, length)
}
