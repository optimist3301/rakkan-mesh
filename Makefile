# Build rakkan-node, and rakkan-web (a web page for a running node), for
# the host or for every release platform.
#
#   make              build ./rakkan-node and ./rakkan-web for this machine
#   make randomx      build librandomx for this machine (build/randomx/)
#   make test         run the tests
#   make dist         cross-compile archives and SHA256SUMS into dist/
#   make dist VERSION=0.013-alpha
#
# RandomX is the reference C++ library, linked through cgo, so a node with it
# needs a C/C++ compiler and cmake (scripts/build-randomx.sh says which for
# each platform). The platforms in RANDOMX_PLATFORMS get it; the others are
# built without cgo and have no RandomX tasks. rakkan-web has no cgo.

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null | sed 's/^v//' || echo dev)
PLATFORMS ?= linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64
RANDOMX_PLATFORMS ?= linux/amd64 linux/arm64 windows/amd64
LDFLAGS := -s -w -X main.version=$(VERSION)
# cgo is on only for RandomX: keep Go's own DNS resolver and user lookup.
TAGS := netgo,osusergo
GO ?= go
HOST := $(shell $(GO) env GOOS GOARCH | tr '\n' '/' | sed 's|/$$||')

.PHONY: all build randomx test dist clean

all: build

# $(call has_randomx,os/arch) is non-empty for a platform with RandomX.
has_randomx = $(filter $(1),$(RANDOMX_PLATFORMS))

randomx:
	@if [ -n "$(call has_randomx,$(HOST))" ]; then sh scripts/build-randomx.sh $(subst /, ,$(HOST)); \
	else echo "no RandomX for $(HOST)"; fi

build: randomx
	@cgo=0; [ -n "$(call has_randomx,$(HOST))" ] && cgo=1; \
	CGO_ENABLED=$$cgo $(GO) build -tags $(TAGS) -trimpath -ldflags '$(LDFLAGS)' -o rakkan-node ./cmd/rakkan-node && \
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o rakkan-web ./cmd/rakkan-web

test: randomx
	@cgo=0; [ -n "$(call has_randomx,$(HOST))" ] && cgo=1; \
	CGO_ENABLED=$$cgo $(GO) vet -tags $(TAGS) ./... && CGO_ENABLED=$$cgo $(GO) test -tags $(TAGS) ./...

dist: clean
	@mkdir -p dist
	@set -e; for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; \
	  name=rakkan-node-$(VERSION)-$$os-$$arch; \
	  ext=; [ $$os = windows ] && ext=.exe; \
	  echo "build $$name"; \
	  mkdir -p dist/$$name; \
	  case " $(RANDOMX_PLATFORMS) " in *" $$os/$$arch "*) rx=1;; *) rx=0;; esac; \
	  if [ $$rx = 1 ]; then \
	    sh scripts/build-randomx.sh $$os $$arch; \
	    env $$(sh scripts/build-randomx.sh env $$os $$arch) CGO_ENABLED=1 GOOS=$$os GOARCH=$$arch $(GO) build -tags $(TAGS) -trimpath -ldflags '$(LDFLAGS)' -o dist/$$name/rakkan-node$$ext ./cmd/rakkan-node; \
	  else \
	    CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch $(GO) build -tags $(TAGS) -trimpath -ldflags '$(LDFLAGS)' -o dist/$$name/rakkan-node$$ext ./cmd/rakkan-node; \
	  fi; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o dist/$$name/rakkan-web$$ext ./cmd/rakkan-web; \
	  cp README.md dist/$$name/; \
	  if [ $$rx = 1 ]; then mkdir -p dist/$$name/licenses/randomx; cp build/randomx/$$os-$$arch/LICENSE dist/$$name/licenses/randomx/; fi; \
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
	rm -rf dist rakkan-node rakkan-web
