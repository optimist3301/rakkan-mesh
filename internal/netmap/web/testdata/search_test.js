// Run by search_test.go: node search_test.js <path to search.js>
const S = require(process.argv[2]);
const assert = require("assert");

const now = Date.UTC(2026, 10, 12, 12, 0, 0);
const c = (v) => [{ value: v, conf: 1000 }];
const graph = {
  networks: [{ id: "nhome", claims: { prefix: c("192.168.1.0/24"), "wifi.ssid": c("Home WiFi"), gateway: c("192.168.1.1") } }],
  devices: [
    { id: "dgw", label: "192.168.1.1", idents: ["lmac:nhome/00:00:5e:00:01:01", "ip:nhome/192.168.1.1"],
      addrs: [{ ip: "192.168.1.1" }], networks: ["nhome"], claims: { role: c("gateway"), port: c("tcp/53") },
      modes: ["local"], methods: ["route"], observers: ["1dd06eb0d98a"], last: now - 60e3,
      class: { kind: "network", conf: 0.95 } },
    { id: "dmac", label: "MacBook-Air.local", idents: ["mac:a4:83:e7:11:22:33", "mdns:nhome/macbook-air.local"],
      addrs: [{ ip: "192.168.1.20" }, { ip: "fe80::1c2d:3e4f" }], networks: ["nhome"],
      claims: { model: c("Mac14,2"), service: c("mdns/_ssh._tcp:22"), "mdns.name": c("MacBook-Air.local"), "dhcp.vendor": c("dhcpcd") },
      modes: ["passive"], methods: ["mdns"], observers: ["1dd06eb0d98a"], last: now - 3 * 3600e3,
      class: { kind: "laptop", conf: 0.92 }, location: { level: "area", logical: "HQ/2/R201" } },
    { id: "dtv", label: "Living Room TV", idents: ["uuid:1234"], addrs: [{ ip: "192.168.1.30" }], networks: ["nhome"],
      claims: { name: c("Living Room TV"), port: c("tcp/8009") }, modes: ["active"], methods: ["ssdp"],
      observers: ["ab12"], last: now - 10 * 86400e3, class: { kind: "tv", conf: 0.8 } },
    { id: "dpub", label: "9.9.9.9", idents: ["gip:9.9.9.9"], addrs: [{ ip: "9.9.9.9" }], networks: [],
      claims: {}, modes: ["active"], methods: ["trace"], observers: ["1dd06eb0d98a"], last: now,
      as: { asn: 19281, name: "QUAD9", country: "US" } },
  ],
  links: [{ a: "dmac", b: "dgw", kind: "wifi", detail: "" }, { a: "dgw", b: "dpub", kind: "l3", detail: "ttl 2" }],
};

function find(q) {
  const ctx = S.context(graph, now);
  const p = S.parse(q, ctx.claimKeys);
  return graph.devices.filter((d) => S.matches(d, p, ctx)).map((d) => d.id).sort().join(",");
}
const eq = (q, want) => assert.strictEqual(find(q), want, q);

// plain words: own facts and adjacent ones
eq("macbook", "dmac");
eq("home wifi", "dgw,dmac,dtv");          // the network's SSID is a fact next to each member
eq("quad9", "dpub");                      // AS name
eq("ttl", "dgw,dpub");                    // link detail
eq("ноутбук", "dmac");                    // type in Russian
eq('"living room"', "dtv");
eq("", "dgw,dmac,dpub,dtv");

// keys
eq("type:laptop", "dmac");
eq("device-type:computer", "dmac");       // group includes leaves
eq("тип:телев", "dtv");
eq("type:unknown", "dpub");
eq("type:tv,network", "dgw,dtv");
eq("port:22", "dmac");                    // from the service announcement
eq("port:tcp/53", "dgw");
eq("service:ssh", "dmac");
eq("ip:192.168.1.0/24", "dgw,dmac,dtv");
eq("ip:192.168.1.2", "");                 // a full address is exact, not a prefix
eq("ip:192.168.1.20", "dmac");
eq("ip:9.", "dpub");
eq("ip:fe80::", "dmac");
eq("mac:a4:83", "dmac");
eq("mac:=a4:83:e7:11:22:33", "dmac");
eq("mac:=a4:83", "");
eq("name:\"living room\"", "dtv");
eq("model:mac14", "dmac");
eq("role:gateway", "dgw");
eq("ssid:home", "dgw,dmac,dtv");
eq("net:192.168.1.77", "dgw,dmac,dtv");
eq("as:19281", "dpub");
eq("as:AS19281", "dpub");
eq("observer:ab", "dtv");
eq("mode:passive", "dmac");
eq("method:trace", "dpub");
eq("loc:area", "dmac");
eq("loc:hq/2", "dmac");
eq("seen:<1h", "dgw,dpub");
eq("seen:>7d", "dtv");
eq("seen:<4ч", "dgw,dmac,dpub");
eq("has:model", "dmac");
eq("has:as", "dpub");
eq("id:dtv", "dtv");
eq("near:type:network", "dmac,dpub");
eq('near:"9.9.9.9"', "dgw");
eq("dhcp.vendor:dhcpcd", "dmac");         // any claim by key

// negation and combination
eq("-type:network ip:192.168.1.0/24", "dmac,dtv");
eq("!port:22 type:computer", "");
eq("home -tv", "dgw,dmac");

// addresses are not keys
eq("fe80::1c2d:3e4f", "dmac");
eq("a4:83:e7:11:22:33", "dmac");

// warnings
{
  const ctx = S.context(graph, now);
  assert.deepStrictEqual(S.parse("bogus:1", ctx.claimKeys).warnings.length, 1);
  assert.deepStrictEqual(S.parse("ssh:abc", ctx.claimKeys).warnings.length, 0);
  assert.deepStrictEqual(S.parse("port:", ctx.claimKeys).warnings.length, 1);
  assert.deepStrictEqual(S.describe(S.parse("-port:22,80 mac", ctx.claimKeys)), ["не port: 22 или 80", "«mac»"]);
}
console.log("ok");
