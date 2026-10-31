#!/bin/bash
# NAT lab: runs rakkan-node daemons behind emulated NATs, without root.
#
#                     "internet" 80.10.0.0/24
#         +------------------+-----------------------+
#     anchor .10         router A .20             router B .30
#     (-relay)           NAT - LAN 10.1.0.0/24    NAT - LAN 10.2.0.0/24
#                            node a 10.1.0.2          node b 10.2.0.2
#
# Everything lives in network namespaces inside one unprivileged user
# namespace. Public addresses are from a routable range on purpose: libp2p
# treats documentation ranges as unroutable. Nothing leaves the sandbox.
#
# usage: natlab.sh <rakkan-node binary> [cone|symmetric] [cone|symmetric] [seconds]
# Extra daemon flags for the NATed nodes can be passed in NODE_FLAGS.
set -euo pipefail

if [[ "${NATLAB_INNER:-}" != 1 ]]; then
	exec env NATLAB_INNER=1 unshare --user --map-root-user --net --mount "$0" "$@"
fi

BIN=$(realpath "$1"); NAT_A=${2:-cone}; NAT_B=${3:-cone}; WAIT=${4:-20}
LAB=$(mktemp -d /tmp/natlab.XXXXXX)
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

# Private /run so that "ip netns" works inside the user namespace.
mount -t tmpfs tmpfs /run && mkdir -p /run/netns
ip link set lo up

n=0
bridge() { ip link add "$1" type bridge && ip link set "$1" up; }
host() { ip netns add "$1" && ip -n "$1" link set lo up; }
# attach <netns> <bridge> <dev> <address/prefix>
attach() {
	n=$((n+1))
	ip link add "v$n" type veth peer name "$3" netns "$1"
	ip link set "v$n" master "$2" up
	ip -n "$1" addr add "$4" dev "$3"
	ip -n "$1" link set "$3" up
}
# router <netns> <public/prefix> <lan bridge> <lan gw/prefix> <cone|symmetric>
router() {
	host "$1"
	attach "$1" inet wan "$2"
	attach "$1" "$3" lan "$4"
	ip netns exec "$1" sysctl -qw net.ipv4.ip_forward=1
	local masq="masquerade"
	[[ "$5" == symmetric ]] && masq="masquerade random"
	ip netns exec "$1" nft -f - <<-NFT
		table ip nat {
			chain post { type nat hook postrouting priority 100; oifname "wan" $masq; }
		}
		table ip filter {
			# Like consumer routers, silently drop unsolicited inbound
			# packets instead of answering with ICMP port unreachable.
			chain input {
				type filter hook input priority 0; policy accept;
				iifname "wan" ct state new drop
			}
			chain forwarding {
				type filter hook forward priority 0; policy drop;
				ct state established,related accept
				iifname "lan" oifname "wan" accept
			}
		}
	NFT
}

bridge inet; bridge lanA; bridge lanB
host anchor; attach anchor inet eth0 80.10.0.10/24
router rA 80.10.0.20/24 lanA 10.1.0.1/24 "$NAT_A"
router rB 80.10.0.30/24 lanB 10.2.0.1/24 "$NAT_B"
host a; attach a lanA eth0 10.1.0.2/24; ip -n a route add default via 10.1.0.1
host b; attach b lanB eth0 10.2.0.2/24; ip -n b route add default via 10.2.0.1

cd "$LAB"
cp "$BIN" ./rakkan-node
NET=$(./rakkan-node network init -out root.key 2>/dev/null)
declare -A ID
for h in anchor a b; do
	ID[$h]=$(./rakkan-node id -key "$h.key")
	./rakkan-node network issue -peer "${ID[$h]}" -out "$h.cert" 2>/dev/null
done

daemon() {
	local h=$1; shift
	ip netns exec "$h" ./rakkan-node daemon -network "$NET" -key "$h.key" -cert "$h.cert" \
		-peers "$h.peers.json" -mdns=false -status 5s "$@" >> "$h.log" 2>&1 &
	PIDS+=($!)
	PID[$h]=$!
}
declare -A PID
daemon anchor -relay -listen /ip4/0.0.0.0/tcp/4001,/ip4/0.0.0.0/udp/4001/quic-v1
sleep 0.5
BOOT=/ip4/80.10.0.10/udp/4001/quic-v1/p2p/${ID[anchor]}
# shellcheck disable=SC2086
daemon a -bootstrap "$BOOT" -listen /ip4/0.0.0.0/tcp/4001,/ip4/0.0.0.0/udp/4001/quic-v1 ${NODE_FLAGS:-}
# shellcheck disable=SC2086
daemon b -bootstrap "$BOOT" -listen /ip4/0.0.0.0/tcp/4001,/ip4/0.0.0.0/udp/4001/quic-v1 ${NODE_FLAGS:-}

echo "NAT A: $NAT_A, NAT B: $NAT_B; anchor=${ID[anchor]: -6} a=${ID[a]: -6} b=${ID[b]: -6}; waiting ${WAIT}s"
sleep "$WAIT"

if [[ "${KILL_ANCHOR:-}" == 1 ]]; then
	kill "${PID[anchor]}"
	echo "anchor stopped; waiting 10s"
	sleep 10
fi
if [[ -n "${RESTART:-}" ]]; then
	# The listed nodes come back knowing only the bootstrap address and
	# their peer caches (RESTART_FLAGS='-peers=' disables the cache).
	for h in $RESTART; do kill "${PID[$h]}"; wait "${PID[$h]}" 2>/dev/null || true; done
	sleep 1
	for h in $RESTART; do
		# shellcheck disable=SC2086
		daemon "$h" -bootstrap "$BOOT" -listen /ip4/0.0.0.0/tcp/4001,/ip4/0.0.0.0/udp/4001/quic-v1 ${NODE_FLAGS:-} ${RESTART_FLAGS:-}
	done
	echo "restarted: $RESTART; waiting 15s"
	sleep 15
fi

# Print each node's latest member table.
for h in a b; do
	echo "--- $h:"
	awk '/ members: /{block=""} {block=block $0 "\n"} END{printf "%s", block}' "$h.log" |
		grep -E "members:|^  " | sed -E "s/${ID[anchor]}/anchor/; s/${ID[a]}/a     /; s/${ID[b]}/b     /"
done
echo "logs: $LAB"
