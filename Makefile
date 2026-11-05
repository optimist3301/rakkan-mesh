# Build rakkan-node for the host or for every release platform.
#
#   make              build ./rakkan-node for this machine
#   make test         run the tests
#   make dist         cross-compile archives and SHA256SUMS into dist/
#   make dist VERSION=0.005-alpha
#
# The node has no cgo, so every platform builds from any machine.

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null | sed 's/^v//' || echo dev)
PLATFORMS ?= linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64
LDFLAGS := -s -w -X main.version=$(VERSION)
GO ?= go

.PHONY: all build test dist clean

all: build

build:
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o rakkan-node ./cmd/rakkan-node

test:
	$(GO) vet ./...
	$(GO) test ./...

dist: clean
	@mkdir -p dist
	@set -e; for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; \
	  name=rakkan-node-$(VERSION)-$$os-$$arch; \
	  bin=rakkan-node; [ $$os = windows ] && bin=rakkan-node.exe; \
	  echo "build $$name"; \
	  mkdir -p dist/$$name; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o dist/$$name/$$bin ./cmd/rakkan-node; \
	  cp README.md dist/$$name/; \
	  mkdir -p dist/$$name/licenses/yggdrasil; \
	  cp internal/ygg/NOTICE internal/ygg/licenses/* dist/$$name/licenses/yggdrasil/; \
	  if [ $$os = windows ]; then \
	    (cd dist && zip -qr $$name.zip $$name); \
	  else \
	    tar -C dist -czf dist/$$name.tar.gz $$name; \
	  fi; \
	  rm -rf dist/$$name; \
	done
	cd dist && sha256sum * > SHA256SUMS

clean:
	rm -rf dist rakkan-node
