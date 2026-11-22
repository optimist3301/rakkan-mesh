// Search language for the viewer.
//
//   words              all facts about a device and the facts next to it
//                      (its networks, AS, link ports); every word must match
//   key:value          one kind of fact: type:laptop port:22 ip:10.0.0.0/8
//   key:a,b            any of the values
//   key:=value         exact value instead of a substring
//   -word, -key:value  must not match
//   "two words"        a phrase; works as a value too: name:"living room"
//   any.claim:value    a claim by its key: dhcp.vendor:android
//
// Pure functions over the graph JSON: no DOM, so node can test it.
(function (root) {
  "use strict";

  const CLASS_GROUP = {
    desktop: "computer", laptop: "computer", computer: "computer", server: "computer",
    phone: "mobile", tablet: "mobile", mobile: "mobile",
  };
  const CLASS_NAMES = {
    desktop: ["desktop", "стационарный", "пк", "pc"], laptop: ["laptop", "notebook", "ноутбук"],
    server: ["server", "сервер", "nas"], computer: ["computer", "компьютер"],
    phone: ["phone", "smartphone", "телефон", "смартфон"], tablet: ["tablet", "планшет"],
    mobile: ["mobile", "мобильный"], tv: ["tv", "телевизор", "медиа", "media"],
    iot: ["iot", "интернет вещей", "умный дом", "smart"], printer: ["printer", "принтер", "мфу"],
    network: ["network", "сетевое", "роутер", "router", "switch", "коммутатор", "ap", "точка доступа"],
    unknown: ["unknown", "неизвестен", "неизвестно", "none"],
  };

  const lc = (s) => String(s == null ? "" : s).toLowerCase();
  const claimValues = (d, k) => (d.claims && d.claims[k] ? d.claims[k].map((v) => v.value) : []);
  const anyHas = (list, v, exact) => list.some((x) => (exact ? lc(x) === v : lc(x).includes(v)));

  // ---------- addresses ----------

  function ip4(s) {
    const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(s);
    if (!m) return null;
    const p = m.slice(1).map(Number);
    if (p.some((x) => x > 255)) return null;
    return ((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]) >>> 0;
  }

  // ipMatch: CIDR (10.0.0.0/8), exact address, or a textual prefix ("10.82.").
  function ipMatch(addr, v, exact) {
    addr = lc(addr);
    const cidr = /^([\d.]+)\/(\d{1,2})$/.exec(v);
    if (cidr) {
      const a = ip4(addr), net = ip4(cidr[1]), bits = Number(cidr[2]);
      if (a === null || net === null || bits > 32) return false;
      const mask = bits === 0 ? 0 : (~0 << (32 - bits)) >>> 0;
      return ((a & mask) >>> 0) === ((net & mask) >>> 0);
    }
    if (exact || ip4(v) !== null) return addr === v;
    return addr.startsWith(v) || (v.includes(":") && addr.includes(v));
  }

  const addrsOf = (d) => (d.addrs || []).map((a) => a.ip);

  // ---------- durations ----------

  const UNIT = { s: 1e3, m: 60e3, min: 60e3, h: 3600e3, d: 86400e3, w: 7 * 86400e3,
    с: 1e3, м: 60e3, мин: 60e3, ч: 3600e3, д: 86400e3, дн: 86400e3, н: 7 * 86400e3 };

  function duration(s) {
    const m = /^(\d+(?:\.\d+)?)\s*([a-zа-я]+)$/.exec(s);
    if (!m || !(m[2] in UNIT)) return null;
    return Number(m[1]) * UNIT[m[2]];
  }

  // ---------- keys ----------

  // Each key: description (for the help list), and test(d, value, exact, ctx).
  const KEYS = {};
  const key = (names, help, test) => {
    for (const n of names) KEYS[n] = { name: names[0], help, test };
  };

  function classTest(d, v, exact) {
    const kind = d.class ? d.class.kind : "unknown";
    const wanted = Object.keys(CLASS_NAMES).filter((k) =>
      CLASS_NAMES[k].some((n) => (exact ? n === v : n.startsWith(v))));
    return wanted.some((w) => w === kind || CLASS_GROUP[kind] === w);
  }

  key(["type", "device-type", "class", "тип"], "тип устройства: laptop, phone, tv, iot, printer, network…", classTest);
  key(["port", "порт"], "открытый порт или порт сервиса: port:22, port:tcp/443", (d, v) => {
    const n = v.replace(/^(tcp|udp)\//, "");
    if (claimValues(d, "port").some((p) => lc(p) === v || p.endsWith("/" + n))) return true;
    return claimValues(d, "service").some((s) => s.endsWith(":" + n));
  });
  key(["service", "svc", "сервис"], "объявленный сервис: service:ipp, svc:_ssh._tcp",
    (d, v, exact) => anyHas(claimValues(d, "service").map((s) => s.replace(/^mdns\//, "")), v, exact));
  key(["ip", "addr", "адрес"], "адрес, префикс или подсеть: ip:10.82.0.0/16, ip:192.168.",
    (d, v, exact) => addrsOf(d).some((a) => ipMatch(a, v, exact)));
  key(["mac"], "MAC-адрес или его часть: mac:00:11:32",
    (d, v, exact) => {
      const m = v.replace(/-/g, ":");
      return d.idents.some((i) => /^l?mac:/.test(i) && (exact ? i === "mac:" + m || i.endsWith("/" + m) : i.includes(m)));
    });
  key(["name", "имя"], "имя: из mDNS, DHCP, UPnP, SNMP",
    (d, v, exact) => anyHas([d.label, ...["name", "hostname", "mdns.name", "mdns.instance", "rdns"].flatMap((k) => claimValues(d, k))], v, exact));
  key(["vendor", "производитель"], "производитель", (d, v, exact) => anyHas(claimValues(d, "vendor"), v, exact));
  key(["model", "модель"], "модель", (d, v, exact) => anyHas(claimValues(d, "model"), v, exact));
  key(["os", "ос"], "операционная система, sysDescr",
    (d, v, exact) => anyHas(["os", "os.desc", "os.kernel"].flatMap((k) => claimValues(d, k)), v, exact));
  key(["role", "роль"], "роль: gateway, router, switch, access-point, observer",
    (d, v, exact) => anyHas(claimValues(d, "role"), v, exact) || (d.is_observer && "observer".startsWith(v)));
  key(["ssid"], "имя Wi-Fi сети: точки доступа и устройства в ней", (d, v, exact, ctx) =>
    anyHas([...claimValues(d, "ssid"), ...claimValues(d, "wifi.ssid"),
      ...d.networks.flatMap((n) => netFacts(ctx, n, ["wifi.ssid"]))], v, exact));
  key(["net", "network", "сеть"], "сеть: id, префикс или SSID", (d, v, exact, ctx) =>
    // net:10.82.5.5 also finds the network containing that address
    d.networks.some((n) => anyHas([n, ...netFacts(ctx, n, ["prefix", "wifi.ssid"])], v, exact) ||
      ipInPrefix(v, netFacts(ctx, n, ["prefix"]))));
  key(["as", "asn"], "автономная система: номер или имя", (d, v, exact) => {
    if (!d.as) return false;
    const num = v.replace(/^as/, "");
    return String(d.as.asn) === num || anyHas([d.as.name, d.as.country], v, exact);
  });
  key(["observer", "наблюдатель"], "кто видел: префикс ключа наблюдателя",
    (d, v) => (d.observers || []).some((o) => lc(o).startsWith(v)));
  key(["mode", "режим"], "как узнали: active, passive, local", (d, v, exact) => anyHas(d.modes || [], v, exact));
  key(["method", "метод"], "метод сбора: mdns, ssdp, lldp, snmp, trace…", (d, v, exact) => anyHas(d.methods || [], v, exact));
  key(["loc", "location", "место"], "место: уровень (precise, area, region, unknown) или название",
    (d, v, exact) => {
      const l = d.location || { level: "unknown" };
      return anyHas([l.level, l.logical, l.region].filter(Boolean), v, exact);
    });
  key(["seen", "видели"], "когда видели: seen:<1h — за последний час, seen:>7d — давно", (d, v, exact, ctx) => {
    const m = /^([<>])?(.+)$/.exec(v);
    const ms = duration(m[2]);
    if (ms === null) return false;
    const age = ctx.now - d.last;
    return m[1] === ">" ? age > ms : age <= ms;
  });
  key(["id"], "id устройства или идентификатор: id:dc6z, id:ssh:",
    (d, v, exact) => (exact ? d.id === v || d.idents.includes(v) : lc(d.id).startsWith(v) || anyHas(d.idents, v, false)));
  key(["has", "есть"], "есть факт с таким ключом: has:loc.geo, has:ssh.banner",
    (d, v) => Object.keys(d.claims || {}).some((k) => k === v || k.startsWith(v + ".")) ||
      (v === "location" && d.location && d.location.level !== "unknown") || (v === "as" && !!d.as));
  key(["near", "рядом"], "связан с устройством, подходящим под запрос: near:10.82.0.1, near:\"type:network\"",
    (d, v, exact, ctx) => (ctx.adj.get(d.id) || []).some((o) => {
      const od = ctx.devices.get(o);
      return od && matches(od, parse(v), Object.assign({}, ctx, { depth: (ctx.depth || 0) + 1 }));
    }));

  function netFacts(ctx, id, keys) {
    const n = ctx.networks.get(id);
    return n ? keys.flatMap((k) => claimValues(n, k)) : [];
  }

  function ipInPrefix(v, prefixes) {
    return ip4(v) !== null && prefixes.some((p) => ipMatch(v, p, false));
  }

  // ---------- parsing ----------

  // tokenize keeps quoted parts together: name:"living room" is one token.
  function tokenize(q) {
    const out = [];
    let cur = "", quoted = false, had = false;
    for (const ch of q) {
      if (ch === '"') { quoted = !quoted; had = true; continue; }
      if (!quoted && /\s/.test(ch)) {
        if (cur || had) out.push(cur);
        cur = ""; had = false;
        continue;
      }
      cur += ch;
    }
    if (cur || had) out.push(cur);
    return out;
  }

  // parse turns a query into terms; unknown keys become claim-key lookups
  // when such a claim exists, otherwise plain text (with a warning).
  function parse(q, claimKeys) {
    const terms = [], warnings = [];
    for (let tok of tokenize(String(q || "").trim())) {
      let neg = false;
      if ((tok.startsWith("-") || tok.startsWith("!")) && tok.length > 1) { neg = true; tok = tok.slice(1); }
      const m = /^([a-zа-яё][\w.\-а-яё]*):(.*)$/i.exec(tok);
      if (m && !looksLikeAddress(tok)) {
        const k = lc(m[1]);
        let v = m[2];
        const exact = v.startsWith("=");
        if (exact) v = v.slice(1);
        const values = v.split(",").map((s) => lc(s.trim())).filter(Boolean);
        if (!values.length) { warnings.push(`пустое значение у «${k}:»`); continue; }
        if (KEYS[k]) { terms.push({ key: KEYS[k].name, values, exact, neg }); continue; }
        if (!claimKeys || claimKeys.has(k)) { terms.push({ claim: k, values, exact, neg }); continue; }
        if (!IDENT_KINDS.has(k)) warnings.push(`неизвестный ключ «${k}:» — ищу как текст`);
      }
      terms.push({ text: lc(tok), neg });
    }
    return { terms, warnings };
  }

  // identifier kinds (spec §3): "ssh:…" typed as is is a text search, not a typo
  const IDENT_KINDS = new Set(["lmac", "rip", "gip", "ssh", "uuid", "rkm", "mdns", "host", "bssid", "lldp"]);

  // looksLikeAddress keeps IPv6 and MAC literals ("fe80::1", "00:11:22:…")
  // from being read as key:value.
  function looksLikeAddress(tok) {
    return /^[0-9a-f]{1,4}(:[0-9a-f]{0,4}){2,}(\/\d+)?$/i.test(tok) || /^[0-9a-f]{2}(-[0-9a-f]{2}){2,}$/i.test(tok);
  }

  // ---------- matching ----------

  // facts is the default haystack: the device's own facts and those next to it.
  function facts(d, ctx) {
    if (d._facts) return d._facts;
    const parts = [d.label, d.id, ...d.idents, ...addrsOf(d), ...(d.modes || []), ...(d.methods || [])];
    for (const k of Object.keys(d.claims || {})) for (const v of d.claims[k]) parts.push(v.value);
    if (d.location) parts.push(d.location.level, d.location.logical, d.location.region);
    if (d.class) parts.push(d.class.kind, ...(CLASS_NAMES[d.class.kind] || []));
    if (d.is_observer) parts.push("observer", "наблюдатель");
    if (d.as) parts.push("as" + d.as.asn, d.as.name, d.as.country);
    for (const n of d.networks) parts.push(n, ...netFacts(ctx, n, ["prefix", "wifi.ssid", "gateway"]));
    for (const l of ctx.linksOf.get(d.id) || []) parts.push(l.kind, l.detail);
    d._facts = parts.filter(Boolean).join("\n").toLowerCase();
    return d._facts;
  }

  function termMatches(d, t, ctx) {
    if (t.text !== undefined) return facts(d, ctx).includes(t.text);
    if (t.claim) return t.values.some((v) => anyHas(claimValues(d, t.claim), v, t.exact));
    if (t.key === "near" && (ctx.depth || 0) > 1) return false; // near:near:… stops here
    const k = KEYS[t.key];
    return t.values.some((v) => k.test(d, v, t.exact, ctx));
  }

  function matches(d, parsed, ctx) {
    return parsed.terms.every((t) => termMatches(d, t, ctx) !== t.neg);
  }

  // context indexes the graph once per render.
  function context(graph, now) {
    const devices = new Map(graph.devices.map((d) => [d.id, d]));
    const networks = new Map(graph.networks.map((n) => [n.id, n]));
    const adj = new Map(), linksOf = new Map();
    const push = (m, k, v) => { if (!m.has(k)) m.set(k, []); m.get(k).push(v); };
    for (const l of graph.links || []) {
      push(adj, l.a, l.b); push(adj, l.b, l.a);
      push(linksOf, l.a, l); push(linksOf, l.b, l);
    }
    const claimKeys = new Set();
    for (const d of graph.devices) for (const k of Object.keys(d.claims || {})) claimKeys.add(k);
    return { devices, networks, adj, linksOf, claimKeys, now: now || Date.now() };
  }

  // describe renders parsed terms for the hint line under the search field.
  function describe(parsed) {
    return parsed.terms.map((t) => {
      const what = t.text !== undefined ? `«${t.text}»` : `${t.key || t.claim}${t.exact ? " =" : ":"} ${t.values.join(" или ")}`;
      return (t.neg ? "не " : "") + what;
    });
  }

  // keyHelp lists the keys once each, for the help popover.
  function keyHelp() {
    const seen = new Set(), out = [];
    for (const [alias, k] of Object.entries(KEYS)) {
      if (seen.has(k.name)) { out.find((o) => o.key === k.name).aliases.push(alias); continue; }
      seen.add(k.name);
      out.push({ key: k.name, help: k.help, aliases: [] });
    }
    return out;
  }

  const api = { parse, matches, context, describe, keyHelp, tokenize, ipMatch, duration };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RakkanSearch = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
