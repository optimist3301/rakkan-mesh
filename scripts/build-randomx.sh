#!/bin/sh
# Builds librandomx, the C++ RandomX library the node links through cgo, as a
# static library for one platform:
#
#   scripts/build-randomx.sh GOOS GOARCH     installs build/randomx/GOOS-GOARCH/{include,lib}
#   scripts/build-randomx.sh env GOOS GOARCH prints the CC and CXX to cgo-build GOOS/GOARCH with
#
# Platforms: linux/amd64, linux/arm64, windows/amd64. Cross compilers: for
# linux/arm64 gcc-aarch64-linux-gnu, for windows/amd64 from Linux
# gcc-mingw-w64-x86-64 (on Windows itself, MSYS2's MinGW-w64 gcc). Needs cmake,
# make and git.
#
# The source is pinned to a commit, not just a tag.
set -eu

TAG=v1.2.1
COMMIT=102f8acf90a7649ada410de5499a7ec62e49e1da
URL=https://github.com/tevador/RandomX

root=$(cd "$(dirname "$0")/.." && pwd)

die() { echo "build-randomx: $*" >&2; exit 1; }

host_os() {
	case "$(uname -s)" in
	Linux) echo linux ;;
	MINGW* | MSYS* | CYGWIN*) echo windows ;;
	*) echo other ;;
	esac
}

host_arch() {
	case "$(uname -m)" in
	x86_64 | amd64) echo amd64 ;;
	aarch64 | arm64) echo arm64 ;;
	*) echo other ;;
	esac
}

# toolchain GOOS GOARCH sets CC, CXX and the cmake system name and processor.
toolchain() {
	os=$1 arch=$2
	case "$os-$arch" in
	linux-amd64) sysname=Linux proc=x86_64 cross=x86_64-linux-gnu- ;;
	linux-arm64) sysname=Linux proc=aarch64 cross=aarch64-linux-gnu- ;;
	windows-amd64) sysname=Windows proc=x86_64 cross=x86_64-w64-mingw32- ;;
	*) die "no RandomX for $os/$arch (linux/amd64, linux/arm64, windows/amd64 only)" ;;
	esac
	if [ "$os-$arch" = "$(host_os)-$(host_arch)" ]; then
		CC=${CC:-gcc} CXX=${CXX:-g++}
	else
		CC=${CC:-${cross}gcc} CXX=${CXX:-${cross}g++}
		command -v "$CC" >/dev/null || die "cross compiler $CC not found"
	fi
}

if [ "${1:-}" = env ]; then
	toolchain "$2" "$3"
	echo "CC=$CC CXX=$CXX"
	exit 0
fi

[ $# -eq 2 ] || die "usage: $0 GOOS GOARCH | env GOOS GOARCH"
os=$1 arch=$2
out=$root/build/randomx/$os-$arch
if [ -f "$out/lib/librandomx.a" ] && [ -f "$out/include/randomx.h" ]; then
	exit 0
fi
toolchain "$os" "$arch"
command -v cmake >/dev/null || die "cmake not found"

# RANDOMX_SRC points at an existing checkout (for offline builds).
src=${RANDOMX_SRC:-$root/build/src/RandomX}
if [ ! -d "$src/.git" ]; then
	git clone -q --branch "$TAG" "$URL" "$src"
fi
[ "$(git -C "$src" rev-parse HEAD)" = "$COMMIT" ] || die "$src is not RandomX $TAG ($COMMIT)"

tmp=$root/build/randomx-obj/$os-$arch
rm -rf "$tmp"
mkdir -p "$tmp"
# Not -DARCH=native: the binary must run on other CPUs. The library picks the
# fastest code (JIT, AES, Argon2) for the CPU it finds at run time.
cmake -S "$src" -B "$tmp" -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_SYSTEM_NAME="$sysname" -DCMAKE_SYSTEM_PROCESSOR="$proc" \
	-DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
	-DCMAKE_INSTALL_PREFIX="$out" -DCMAKE_INSTALL_LIBDIR=lib \
	-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
cmake --build "$tmp" --target randomx -j "$(nproc 2>/dev/null || echo 2)" >/dev/null
cmake --install "$tmp" >/dev/null
cp "$src/LICENSE" "$out/LICENSE"
rm -rf "$tmp"
echo "librandomx $TAG for $os/$arch: $out"
