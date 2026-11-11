#!/bin/sh
# Builds a fully static rakkan-node for Linux, linked against musl in an
# Alpine container, so it runs on any distribution whatever its C library
# (glibc of any age, or musl on Alpine):
#
#   scripts/build-linux-static.sh ARCH OUT VERSION   writes OUT/rakkan-node
#
# ARCH is amd64 or arm64; another architecture than the host's runs under
# QEMU emulation (binfmt_misc, e.g. qemu-user-static), which is slow.
# Needs docker or podman (CONTAINER picks one). RANDOMX_SRC may point at a
# RandomX checkout to build from instead of cloning it.
set -eu

GO_IMAGE=docker.io/library/golang:1.26.10-alpine

[ $# -eq 3 ] || { echo "usage: $0 ARCH OUT VERSION" >&2; exit 2; }
arch=$1 out=$2 version=$3
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)

engine=${CONTAINER:-}
if [ -z "$engine" ]; then
	for e in docker podman; do
		if command -v $e >/dev/null && $e info >/dev/null 2>&1; then engine=$e; break; fi
	done
fi
[ -n "$engine" ] || { echo "build-linux-static: needs docker or podman" >&2; exit 1; }

# The musl library and objects live apart from the host's build/ (which may
# hold a glibc librandomx): /src/build in the container is build/musl-ARCH.
mkdir -p "$root/build/musl-$arch"
set -- -v "$root:/src:ro" -v "$root/build/musl-$arch:/src/build" -v "$out:/out"
if [ -n "${RANDOMX_SRC:-}" ]; then
	set -- "$@" -v "$RANDOMX_SRC:/randomx-src:ro" -e RANDOMX_SRC=/randomx-src
fi
modcache=$(go env GOMODCACHE 2>/dev/null || true)
if [ -n "$modcache" ] && [ -d "$modcache" ]; then
	set -- "$@" -v "$modcache:/go/pkg/mod"
fi

# label=disable: on SELinux hosts the mounts are readable without relabeling
# the checkout.
exec "$engine" run --rm --platform "linux/$arch" --security-opt label=disable "$@" -e VERSION="$version" -w /src "$GO_IMAGE" sh -euc '
	apk add --no-cache build-base cmake git linux-headers >/dev/null
	git config --global --add safe.directory "*"
	sh scripts/build-randomx.sh linux $(go env GOARCH)
	CGO_ENABLED=1 go build -buildvcs=false -tags netgo,osusergo -trimpath \
		-ldflags "-s -w -X main.version=$VERSION -linkmode external -extldflags -static" \
		-o /out/rakkan-node ./cmd/rakkan-node
	/out/rakkan-node version
'
