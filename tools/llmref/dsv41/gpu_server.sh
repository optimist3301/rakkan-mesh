#!/usr/bin/env bash
# Checks to run on a rented NVIDIA machine (Hopper, e.g. one H100), in one
# go, results in one archive to bring back:
#
#   1. the CPU stand-in of the reference's kernels (kernel_cpu.py) against
#      the real tilelang kernels (check_kernels.py), on the tiny fixture and
#      on a medium one: this decides whether kernel_cpu needs changes;
#   2. the Go engine's tests, the Vulkan ones on the NVIDIA GPU if its
#      driver offers Vulkan: the same bits as the CPU on another vendor;
#   3. with --real: the published layers 2 and 20 against the reference
#      (16 GiB downloaded from the Hub), on the CPU and the GPU.
#
# usage, from the repository copied to the machine:
#   tools/llmref/dsv41/gpu_server.sh [--real]
# Environment: TORCH_CUDA (the PyTorch wheel index, default: tried
# cu132, cu130, cu128, cu126), GO_VERSION (default from go.mod), OUT (results
# directory). Every step logs to OUT/<step>.log; one failing does not stop
# the others. The archive is OUT.tar.gz.
#
# Rented machines often have a small disk (20 GB) and a large /dev/shm: the
# caches (pip, tilelang's kernels, Go's, the Hub shards of --real) go to
# /dev/shm when it has room, the repository may stay on the disk.
set -uo pipefail

if [ -d /dev/shm ] && [ "$(df --output=avail -k /dev/shm | tail -1)" -gt $((40 << 20)) ]; then
	export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/cache}
	export GOCACHE=${GOCACHE:-/dev/shm/go/cache} GOMODCACHE=${GOMODCACHE:-/dev/shm/go/mod}
fi
export PIP_NO_CACHE_DIR=1

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$REPO/gpu-results-$(date +%Y%m%d-%H%M%S)}
REAL=0
[ "${1:-}" = "--real" ] && REAL=1
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
VENV=$HERE/../.venv-gpu
PY=$VENV/bin/python

say() { echo "$*" | tee -a "$SUMMARY"; }

# step NAME COMMAND...: runs a step, logged, and notes how it went.
step() {
	local name=$1
	shift
	local start=$SECONDS
	echo "== $name"
	# A subshell: a step's set -e or -x stays its own.
	if ("$@") >"$OUT/$name.log" 2>&1; then
		say "ok      $name ($((SECONDS - start)) s)"
	else
		say "FAILED  $name ($((SECONDS - start)) s): see $name.log"
	fi
}

# What a bare image lacks (Ubuntu): venv, and the Vulkan loader with the
# libraries NVIDIA's Vulkan driver loads. Root and apt only.
packages() {
	set -e
	[ "$(id -u)" = 0 ] && command -v apt-get >/dev/null || { echo "not root or no apt: skipped"; return 0; }
	export DEBIAN_FRONTEND=noninteractive
	apt-get update
	apt-get install -y python3-venv "python$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')-venv" \
		libvulkan1 vulkan-tools libxext6 libx11-6 libegl1 libgl1 curl git
}

say "rakkan-mesh GPU checks, $(date -u +%FT%TZ), revision $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

git config --global --add safe.directory "$REPO" # copied in as another user
step packages packages

# 0. The machine.
environment() {
	set -x
	nvidia-smi
	nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
	nvcc --version || true
	uname -a
	nproc
	free -g
	df -h "$REPO"
	python3 --version
	vulkaninfo --summary || echo "no vulkaninfo: install vulkan-tools (and the driver's Vulkan ICD, libnvidia-gl-*) for the Vulkan tests"
}
step environment environment

# 1. Python with CUDA PyTorch (the fixture's version) and tilelang.
python_env() {
	set -e
	python3 -m venv "$VENV"
	"$PY" -m pip install --upgrade pip
	local tags=${TORCH_CUDA:-cu132 cu130 cu128 cu126}
	local ok=0
	for tag in $tags; do
		if "$PY" -m pip install --index-url "https://download.pytorch.org/whl/$tag" torch==2.15.0; then
			ok=1
			break
		fi
	done
	[ $ok = 1 ]
	"$PY" -m pip install -r "$HERE/requirements-cpu.txt" tilelang
	# tilelang compiles with nvcc: the pip one, of torch's CUDA version (a
	# newer nvcc makes PTX the older ptxas rejects, and wants its headers).
	local cu
	cu=$("$PY" -c 'import torch; print(torch.version.cuda)')
	"$PY" -m pip install "nvidia-cuda-nvcc==$cu.*" "nvidia-cuda-cccl==$cu.*" "nvidia-cuda-crt==$cu.*" "nvidia-nvvm==$cu.*"
	"$PY" - <<-'EOF'
		import torch, tilelang
		print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
		print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
		print("tilelang", tilelang.__version__)
		assert torch.cuda.is_available()
	EOF
}
step python-env python_env

in_here() { (cd "$HERE" && "$@"); }

# 2. The kernels: kernel_cpu's own check, then against tilelang.
step kernel-cpu-self in_here "$PY" test_kernel_cpu.py
step kernels-tiny in_here "$PY" check_kernels.py --out "$OUT/report-tiny.json"
step medium-fixture in_here "$PY" tiny.py --medium "$OUT/medium"
step kernels-medium in_here "$PY" check_kernels.py --fixture "$OUT/medium" --out "$OUT/report-medium.json"

# 3. Go, the version go.mod asks for.
go_env() {
	set -e
	local want=${GO_VERSION:-$(awk '/^go /{print $2}' "$REPO/go.mod")}
	if ! command -v go >/dev/null || [ "$(go env GOVERSION)" != "go$want" ]; then
		mkdir -p "$HOME/go-sdk"
		curl -fsSL "https://go.dev/dl/go$want.linux-amd64.tar.gz" | tar -xz -C "$HOME/go-sdk"
	fi
	go version
}
export PATH=$HOME/go-sdk/go/bin:$PATH
step go-env go_env

in_repo() { (cd "$REPO" && "$@"); }
step go-gpu-kernels in_repo go test -count=1 -v ./internal/gpu
step go-dsv41 in_repo go test -count=1 -v -timeout 30m ./internal/llm/dsv41
step go-llm in_repo go test -count=1 -timeout 30m ./internal/num ./internal/llm/...

# 4. The real layers (--real): the reference on them, then the engine.
if [ $REAL = 1 ]; then
	CACHE=$(in_here "$PY" -c 'import fetch; print(fetch.default_cache(fetch.pins()["revision"]))')
	step fetch-shards in_here "$PY" fetch.py tokenizer.json model-00002-of-00048.safetensors \
		model-00005-of-00048.safetensors model-00023-of-00048.safetensors
	step reference-layer2 in_here "$PY" layer_check.py 2 "$CACHE/layer2.safetensors"
	step reference-layer20 in_here "$PY" layer_check.py 20 "$CACHE/layer20.safetensors"
	step go-real-layers in_repo env RAKKAN_DSV41_REAL="$CACHE" go test -count=1 -v -timeout 60m -run 'RealLayers' ./internal/llm/dsv41
fi

# What came out of the checks, at a glance.
for r in "$OUT"/report-*.json; do
	[ -f "$r" ] || continue
	say "--- $(basename "$r"): entries below 1.0 identical"
	"$PY" - "$r" >>"$SUMMARY" <<-'EOF' || true
		import json, sys
		def walk(prefix, v):
		    if isinstance(v, dict):
		        if "identical" in v and v["identical"] < 1:
		            print(f"  {prefix}: {v['identical']:.6f} identical, max_abs {v.get('max_abs')}")
		        for k, x in v.items():
		            walk(f"{prefix}.{k}" if prefix else k, x)
		walk("", json.load(open(sys.argv[1])))
	EOF
done
grep -h -E '^(--- (PASS|FAIL|SKIP)|ok |FAIL)' "$OUT"/go-*.log >>"$SUMMARY" 2>/dev/null || true

rm -rf "$OUT/medium"/*.safetensors # large, and remade by tiny.py --medium
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
cat "$SUMMARY"
echo
echo "Bring back: $OUT.tar.gz"
