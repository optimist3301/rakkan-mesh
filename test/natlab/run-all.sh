#!/bin/bash
# Runs the NAT lab scenarios and checks how node a reaches node b.
# usage: run-all.sh <rakkan-node binary>
set -uo pipefail
LAB=$(dirname "$0")/natlab.sh
BIN=$1
fails=0

# check <description> <expected: direct|relayed|connected|none> <env...> -- <natlab args>
check() {
	local desc=$1 want=$2; shift 2
	local envs=()
	while [[ $1 != -- ]]; do envs+=("$1"); shift; done; shift
	local out got
	out=$(env "${envs[@]}" "$LAB" "$BIN" "$@" 2>&1)
	got=$(grep -A3 "^--- a:" <<<"$out" | awk '$1 == "b" {print $3}')
	[[ -z $got ]] && got=none
	if [[ $got == "$want" || ( $want == connected && $got != none ) ]]; then
		printf "ok    %-50s %s\n" "$desc" "$got"
	else
		printf "FAIL  %-50s got %s, want %s\n" "$desc" "$got" "$want"
		fails=$((fails+1))
	fi
}

check "cone <-> cone: hole punched"                 direct  -- cone cone 20
check "cone <-> symmetric: relayed"                  relayed -- cone symmetric 25
check "symmetric <-> symmetric: relayed"             relayed -- symmetric symmetric 25
check "anchor down after connecting"                 direct  KILL_ANCHOR=1 -- cone cone 20
check "both restart, anchor down, peer cache"        direct  KILL_ANCHOR=1 "RESTART=a b" -- cone cone 20
check "both restart, anchor down, no cache"          none    KILL_ANCHOR=1 "RESTART=a b" RESTART_FLAGS=-peers= -- cone cone 20
exit $fails
