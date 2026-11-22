// rakkan map viewer. Every string that comes from map data (names, SSIDs,
// claims) may come from someone else's map: it is only ever put into the
// page as text (textContent), never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  meta: null,
  graph: null,
  at: 0,          // 0 = now (whole set)
  view: "graph",
  selected: null, // {type: "device"|"network", id}
  positions: new Map(),
  mapFitted: false,
};
const byDev = new Map();
const byNet = new Map();

// ---------- helpers ----------

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "onclick") e.addEventListener("click", v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

const best = (claims, k) => (claims && claims[k] && claims[k].length ? claims[k][0].value : "");
const role = (d) => best(d.claims, "role");
const fmtTime = (ms) => new Date(ms).toLocaleString("ru-RU");
const meters = (m) => (m >= 1000 ? (m / 1000).toFixed(1) + " км" : Math.round(m) + " м");
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const pct = (c) => Math.round(c * 100) + "%";

function ago(ms) {
  const s = (Date.now() - ms) / 1000;
  if (s < 90) return "сейчас";
  if (s < 3600) return Math.round(s / 60) + " мин назад";
  if (s < 172800) return Math.round(s / 3600) + " ч назад";
  return Math.round(s / 86400) + " дн назад";
}

function kindOf(d) {
  if (d.is_observer) return "observer";
  const r = role(d);
  // A Thread border router (HomePod, Apple TV) also sends router
  // advertisements: draw it as what it is unless it is network gear.
  if (r === "gateway" || (r === "router" && ["", "network"].includes(classKind(d)))) return "router";
  if (r === "switch") return "switch";
  if (r === "access-point") return "ap";
  return "dev";
}

// Device types (spec §12) and the colour group each is drawn with.
const CLASS_RU = {
  desktop: "стационарный компьютер", laptop: "ноутбук", server: "сервер", computer: "компьютер",
  phone: "телефон", tablet: "планшет", mobile: "телефон или планшет", tv: "телевизор, медиа",
  iot: "интернет вещей", printer: "принтер", network: "сетевое оборудование",
};
const CLASS_GROUP = {
  desktop: "computer", laptop: "computer", computer: "computer", server: "server",
  phone: "mobile", tablet: "mobile", mobile: "mobile", tv: "tv", iot: "iot", printer: "printer", network: "network",
};
const classKind = (d) => (d.class ? d.class.kind : "");
const classGroup = (d) => CLASS_GROUP[classKind(d)] || "unknown";

function classMatches(d, want) {
  const k = classKind(d);
  if (!want) return true;
  if (want === "unknown") return !k;
  if (want === "computer") return CLASS_GROUP[k] === "computer" || k === "server";
  if (want === "mobile") return CLASS_GROUP[k] === "mobile";
  return k === want;
}

function devLabel(d) {
  return d ? d.label : "?";
}

function netLabel(n) {
  const p = best(n.claims, "prefix") || n.id;
  const ssid = best(n.claims, "wifi.ssid");
  return ssid ? `${p} · ${ssid}` : p;
}

// replaceChildren would print null as "null": drop absent parts.
function fill(box, ...parts) { box.replaceChildren(...parts.filter((p) => p != null)); }

function status(text) { $("status").textContent = text || ""; }

// ---------- data ----------

// Embedded in rakkan-web the data comes from its protected /app/ and the
// token is the one the main page keeps.
const API = document.documentElement.dataset.api || "api/";
function authHeaders() {
  try {
    const t = JSON.parse(localStorage.getItem("rakkan.token") || '""');
    return t ? { Authorization: "Bearer " + t } : {};
  } catch { return {}; }
}
if (document.documentElement.dataset.home === "1") {
  $("home").classList.remove("hidden");
  $("brand").classList.add("hidden");
}

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store", headers: authHeaders() });
  if (r.status === 401) throw new Error("нужен токен: введите его на главной странице узла");
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function loadMeta() {
  state.meta = await getJSON(API + "meta");
}

async function loadGraph() {
  status("загрузка…");
  const q = state.at ? `?at=${state.at}` : "";
  const g = await getJSON(API + "graph" + q);
  state.graph = g;
  byDev.clear(); byNet.clear();
  for (const d of g.devices) byDev.set(d.id, d);
  for (const n of g.networks) byNet.set(n.id, n);
  fillNetworks();
  render();
  status("");
}

function fillNetworks() {
  const sel = $("net");
  const cur = sel.value;
  sel.replaceChildren(el("option", { value: "" }, "Все сети"));
  for (const n of state.graph.networks) {
    sel.append(el("option", { value: n.id }, `${netLabel(n)} (${n.devices.length})`));
  }
  sel.value = byNet.has(cur) ? cur : "";
}

// ---------- filters ----------

function filters() {
  return {
    q: searchQuery(),
    net: $("net").value,
    cls: $("cls").value,
    modes: new Set([...document.querySelectorAll(".mode:checked")].map((c) => c.value)),
    kinds: new Set([...document.querySelectorAll(".kind:checked")].map((c) => c.value)),
    onlyLocated: $("only-located").checked,
    hideAP: $("hide-ap").checked,
  };
}

// ---------- search ----------

// The search language is in search.js; this is its UI: the hint line under
// the field (what the query means, warnings, key completion) and the help.
let searchCtx = null;
function searchContext() {
  const now = state.at || Date.now();
  if (!searchCtx || searchCtx.graph !== state.graph || searchCtx.now !== now) {
    searchCtx = Object.assign(RakkanSearch.context(state.graph, now), { graph: state.graph });
  }
  return searchCtx;
}

function searchQuery() {
  const ctx = state.graph ? searchContext() : null;
  return RakkanSearch.parse($("q").value, ctx && ctx.claimKeys);
}

// keySuggestions: keys starting with the word being typed (the last one).
function keySuggestions() {
  const q = $("q").value;
  // only while a bare word is being typed: not inside "seen:<1h" or a quote
  const tok = q.split(/\s+/).pop();
  const last = (/^[-!]?([a-zа-яё][a-zа-яё.\-]*)$/i.exec(tok) || [])[1];
  if (!last || (q.split('"').length - 1) % 2) return [];
  const w = last.toLowerCase();
  const out = [];
  for (const k of RakkanSearch.keyHelp()) {
    const name = [k.key, ...k.aliases].find((a) => a.startsWith(w) && a !== w);
    if (name) out.push({ name, help: k.help });
  }
  if (state.graph) {
    for (const ck of searchContext().claimKeys) {
      if (ck.startsWith(w) && ck !== w && ck.includes(".") && out.length < 8) out.push({ name: ck, help: "утверждение" });
    }
  }
  return out.slice(0, 6);
}

function completeKey(name) {
  const q = $("q");
  q.value = q.value.replace(/([-!]?)([a-zа-яё.\-]+)$/i, (_, neg) => `${neg}${name}:`);
  q.focus();
  updateSearchHint();
  render();
}

function updateSearchHint() {
  const hint = $("q-hint");
  const sugg = keySuggestions();
  if (sugg.length) {
    fill(hint, ...sugg.map((s) => el("button", { class: "q-sugg", type: "button", title: s.help,
      onclick: () => completeKey(s.name) }, s.name + ":")));
    return;
  }
  const p = searchQuery();
  const parts = RakkanSearch.describe(p).filter((_, i) => p.terms[i].text === undefined || p.terms.length > 1);
  fill(hint,
    ...p.warnings.map((w) => el("div", { class: "warn" }, w)),
    parts.length ? el("div", {}, parts.join(" · ")) : null);
}

function buildSearchHelp() {
  const rows = RakkanSearch.keyHelp().map((k) => el("li", {},
    el("button", { class: "q-sugg", type: "button", onclick: () => { completeKey(k.key); toggleSearchHelp(false); } }, k.key + ":"),
    el("span", {}, " " + k.help),
    k.aliases.length ? el("span", { class: "sub" }, ` (${k.aliases.join(", ")})`) : null));
  fill($("q-help"),
    el("p", {}, "Слова ищутся по всем фактам об устройстве и по соседним: его сети, SSID, AS, портам связей. Все слова должны совпасть."),
    el("ul", { class: "q-syntax" },
      el("li", {}, el("code", {}, "ключ:значение"), " — один вид фактов"),
      el("li", {}, el("code", {}, "port:22,80"), " — любое из значений"),
      el("li", {}, el("code", {}, "mac:=…"), " — точное совпадение"),
      el("li", {}, el("code", {}, "-type:tv"), " — исключить"),
      el("li", {}, el("code", {}, 'name:"living room"'), " — фраза"),
      el("li", {}, el("code", {}, "dhcp.vendor:android"), " — любое утверждение по ключу")),
    el("ul", { class: "q-keys" }, ...rows));
}

function toggleSearchHelp(open) {
  const box = $("q-help");
  const show = open === undefined ? box.classList.contains("hidden") : open;
  if (show && !box.firstChild) buildSearchHelp();
  box.classList.toggle("hidden", !show);
  $("q-help-btn").setAttribute("aria-expanded", String(show));
}

function visible(d, f) {
  if (f.q.terms.length && !RakkanSearch.matches(d, f.q, searchContext())) return false;
  if (f.net && !d.networks.includes(f.net)) return false;
  if (!classMatches(d, f.cls)) return false;
  if (!d.modes.some((m) => f.modes.has(m))) return false;
  if (f.onlyLocated && (!d.location || d.location.level === "unknown")) return false;
  if (f.hideAP && kindOf(d) === "ap" && d.networks.length === 0) return false;
  return true;
}

// ---------- graph view ----------

let cy = null;

function cyStyle() {
  const css = getComputedStyle(document.documentElement);
  const v = (n) => css.getPropertyValue(n).trim();
  // Colours and shapes only; sizes are set per zoom level by updateLabels().
  return [
    { selector: "node", style: {
      label: "data(label)", "font-family": v("--font"), "font-weight": 400, color: "#a0a0a8",
      "text-valign": "bottom", "text-wrap": "ellipsis", "text-outline-color": "#000",
      "background-color": v("--c-unknown"), "border-width": 0,
    } },
    // device types first: roles below override them (a gateway is drawn as a gateway)
    ...["computer", "server", "mobile", "tv", "iot", "printer", "network"].map((g) => (
      { selector: `node.c-${g}`, style: { "background-color": v(`--c-${g}`) } })),
    { selector: "node.observer", style: { "background-color": v("--observer"), color: "#fff", "font-weight": 600 } },
    { selector: "node.router", style: { "background-color": v("--router"), shape: "diamond", color: "#fff" } },
    { selector: "node.switch", style: { "background-color": v("--switch"), shape: "round-rectangle", color: "#fff" } },
    { selector: "node.ap", style: { "background-color": v("--ap"), shape: "round-triangle" } },
    { selector: "node.net", style: {
      "background-color": v("--gray4"), shape: "round-rectangle", color: "#fff", "font-weight": 600, "text-valign": "center",
      "z-compound-depth": "top", "z-index": 20,
    } },
    { selector: "node.as", style: {
      "background-color": "#0b1a20", "border-color": v("--cyan"), shape: "round-rectangle", color: v("--cyan"),
      "font-weight": 600, "text-valign": "center", "z-index": 1, // under devices, never hiding their names
    } },
    { selector: "node:selected", style: { "border-color": "#fff", color: "#fff", "font-weight": 600, "z-index": 10 } },
    // Cytoscape ignores the alpha of rgba() colours: opacity is separate.
    { selector: "edge", style: { "line-color": "#ebebf5", "line-opacity": 0.12, "curve-style": "haystack" } },
    { selector: "edge.l2", style: { "line-color": "#ebebf5", "line-opacity": 0.8, "curve-style": "bezier" } },
    { selector: "edge.l2-behind", style: { "line-style": "dashed", "line-color": "#ebebf5", "line-opacity": 0.4, "curve-style": "bezier" } },
    { selector: "edge.l3", style: { "line-color": v("--cyan"), "line-opacity": 0.9, "curve-style": "bezier" } },
    { selector: "edge.wan", style: { "line-color": v("--cyan"), "line-opacity": 0.9, "curve-style": "bezier" } },
    { selector: "edge.as", style: { "line-style": "dashed", "line-color": v("--cyan"), "line-opacity": 0.3, "curve-style": "haystack" } },
    { selector: "edge.wifi", style: { "line-style": "dotted", "line-color": v("--purple"), "line-opacity": 0.9, "curve-style": "bezier" } },
    { selector: "edge:selected", style: { "line-color": v("--blue"), "line-opacity": 1 } },
  ];
}

// Dots, lines and labels keep a constant on-screen size whatever the zoom
// (as on a map): zooming in spreads things apart instead of inflating them.
// Names of ordinary devices appear only once zoomed in enough not to pile up.
const PX = {
  font: { dev: 10, key: 11, net: 12 },
  node: { dev: 7, observer: 12, router: 11, switch: [11, 8], ap: 9 },
  edge: { member: 0.6, l2: 1.5, "l2-behind": 1, l3: 1.5, wifi: 1.5, wan: 1.5, as: 0.7 },
};
const SHOW_DEV_LABELS_AT = 1.1;

function updateLabels() {
  if (!cy) return;
  const z = cy.zoom();
  const showDev = z >= SHOW_DEV_LABELS_AT;
  const n = PX.node;
  cy.batch(() => {
    let st = cy.style()
      .selector("node").style({
        width: n.dev / z, height: n.dev / z,
        "font-size": PX.font.dev / z, "text-margin-y": 4 / z, "text-max-width": 140 / z,
        "text-outline-width": 2 / z, label: showDev ? "data(label)" : "",
      })
      .selector("node.observer").style({ width: n.observer / z, height: n.observer / z })
      .selector("node.router").style({ width: n.router / z, height: n.router / z })
      .selector("node.switch").style({ width: n.switch[0] / z, height: n.switch[1] / z })
      .selector("node.ap").style({ width: n.ap / z, height: n.ap / z })
      .selector("node.observer, node.router, node.switch, node:selected")
      .style({ "font-size": PX.font.key / z, label: "data(label)" })
      .selector("node:selected").style({ "border-width": 2 / z })
      .selector("node.as").style({
        "font-size": PX.font.key / z, label: "data(label)", padding: 7 / z, height: 20 / z,
        width: "label", "text-max-width": 260 / z, "text-margin-y": 0, "text-outline-width": 0, "border-width": 1 / z,
      })
      .selector("node.net").style({
        "font-size": PX.font.net / z, label: "data(label)", padding: 9 / z, height: 24 / z,
        width: "label", "text-max-width": 320 / z, "text-margin-y": 0, "text-outline-width": 0,
      })
      .selector("edge").style({ width: PX.edge.member / z });
    for (const [k, w] of Object.entries(PX.edge)) {
      if (k !== "member") st = st.selector("edge." + k).style({ width: w / z });
    }
    st.selector("edge.l2-behind, edge.as").style({ "line-dash-pattern": [5 / z, 4 / z] })
      .selector("edge.wifi").style({ "line-dash-pattern": [1.5 / z, 3 / z] })
      .update();
  });
}

// Generous spacing: dense networks read better spread out than packed.
const LAYOUT = {
  name: "cose", animate: false, nodeDimensionsIncludeLabels: true, // wide AS and network plates need room
  nodeRepulsion: 120000, idealEdgeLength: 150, edgeElasticity: 60, nodeOverlap: 40,
  gravity: 0.15, componentSpacing: 160, padding: 40,
};

function renderGraph(devs, f) {
  const shown = new Set(devs.map((d) => d.id));
  const els = [];
  const nets = new Set();
  for (const d of devs) for (const n of d.networks) if (!f.net || n === f.net) nets.add(n);
  for (const id of nets) {
    const n = byNet.get(id);
    if (n) els.push({ group: "nodes", data: { id: "n:" + id, label: netLabel(n), net: id }, classes: "net" });
  }
  for (const d of devs) {
    els.push({ group: "nodes", data: { id: d.id, label: trunc(devLabel(d), 28), dev: d.id }, classes: `${kindOf(d)} c-${classGroup(d)}` });
    if (f.kinds.has("member")) {
      for (const n of d.networks) {
        if (nets.has(n)) els.push({ group: "edges", data: { id: `m:${d.id}:${n}`, source: d.id, target: "n:" + n }, classes: "member" });
      }
    }
  }
  if (f.kinds.has("as")) {
    for (const gr of state.graph.ases || []) {
      const devIn = gr.devices.filter((id) => shown.has(id));
      const netIn = (gr.networks || []).filter((id) => nets.has(id));
      if (!devIn.length && !netIn.length) continue;
      const id = "as:" + gr.asn;
      els.push({ group: "nodes", data: { id, label: `AS${gr.asn} ${gr.name}`, as: gr.asn }, classes: "as" });
      for (const d of devIn) els.push({ group: "edges", data: { id: `a:${d}:${id}`, source: d, target: id }, classes: "as" });
      for (const n of netIn) els.push({ group: "edges", data: { id: `a:n:${n}:${id}`, source: "n:" + n, target: id }, classes: "as" });
    }
  }
  for (const l of state.graph.links) {
    if (!f.kinds.has(l.kind) || !shown.has(l.a) || !shown.has(l.b)) continue;
    els.push({ group: "edges", data: { id: `l:${l.a}:${l.b}:${l.kind}:${l.detail || ""}`, source: l.a, target: l.b }, classes: l.kind });
  }

  if (!cy) {
    cy = cytoscape({ container: $("graph"), style: cyStyle(), wheelSensitivity: 0.3, minZoom: 0.05, maxZoom: 6, pixelRatio: "auto" });
    cy.on("tap", "node", (e) => {
      const d = e.target.data();
      if (d.as) { select({ type: "as", id: d.as }); return; }
      select(d.dev ? { type: "device", id: d.dev } : { type: "network", id: d.net });
    });
    cy.on("tap", (e) => { if (e.target === cy) select(null); });
    let zt = null;
    cy.on("zoom", () => { cancelAnimationFrame(zt); zt = requestAnimationFrame(updateLabels); });
    cy.on("select unselect", "node", () => requestAnimationFrame(updateLabels));
  }
  // Keep positions across filter and time changes so the picture is stable.
  cy.nodes().forEach((n) => state.positions.set(n.id(), { ...n.position() }));
  cy.elements().remove();
  cy.add(els);
  const fresh = cy.nodes().filter((n) => !state.positions.has(n.id()));
  cy.nodes().forEach((n) => { const p = state.positions.get(n.id()); if (p) n.position(p); });
  if (fresh.length) {
    const layout = fresh.length === cy.nodes().length
      ? cy.layout({ ...LAYOUT, randomize: true, numIter: 2500 })
      : cy.layout({ ...LAYOUT, randomize: false, numIter: 800 });
    layout.run();
  }
  updateLabels();
  if (state.selected) {
    const sel = cy.getElementById(state.selected.type === "device" ? state.selected.id : "n:" + state.selected.id);
    if (sel.nonempty()) sel.select();
  }
}

// ---------- map view ----------

let map = null, mapLayer = null;
const levelColor = { precise: "--precise", area: "--area", region: "--region" };

function ensureMap() {
  if (map) return;
  map = L.map("map", { worldCopyJump: true, attributionControl: true }).setView([20, 0], 2);
  if (state.meta.tiles) {
    L.tileLayer(state.meta.tiles, {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors",
    }).addTo(map);
  }
  mapLayer = L.layerGroup().addTo(map);
}

function renderMap(devs, f) {
  ensureMap();
  mapLayer.clearLayers();
  const css = getComputedStyle(document.documentElement);
  const located = devs.filter((d) => d.location && d.location.point);
  $("map-empty").classList.toggle("hidden", located.length > 0 || state.view !== "map");
  const pos = new Map();
  const bounds = [];
  for (const d of located) {
    const l = d.location;
    const ll = [l.point.lat, l.point.lon];
    pos.set(d.id, ll);
    bounds.push(ll);
    const color = css.getPropertyValue(levelColor[l.level] || "--region").trim();
    L.circle(ll, { radius: l.radius_m, color, weight: 1, opacity: 0.6, fillColor: color, fillOpacity: 0.12,
      dashArray: l.derived ? "4 4" : null, interactive: false }).addTo(mapLayer);
    const m = L.circleMarker(ll, { radius: d.is_observer ? 7 : 5, color: "#fff", weight: 1.5, fillColor: color, fillOpacity: 1 })
      .addTo(mapLayer);
    // Tooltip content is a DOM node built from text, never an HTML string.
    m.bindTooltip(el("span", {}, `${devLabel(d)} · ${l.level} ±${meters(l.radius_m)}`));
    m.on("click", () => select({ type: "device", id: d.id }));
  }
  for (const lk of state.graph.links) {
    if (!f.kinds.has(lk.kind) || !pos.has(lk.a) || !pos.has(lk.b)) continue;
    L.polyline([pos.get(lk.a), pos.get(lk.b)], { color: "rgba(235,235,245,0.45)", weight: 1.5,
      dashArray: lk.kind === "l2" ? null : "4 4", interactive: false }).addTo(mapLayer);
  }
  if (bounds.length && !state.mapFitted) {
    map.fitBounds(bounds, { maxZoom: 17, padding: [40, 40] });
    state.mapFitted = true;
  }
}

// ---------- details ----------

function devLink(id) {
  const d = byDev.get(id);
  return el("a", { class: "dev", onclick: () => select({ type: "device", id }) }, d ? devLabel(d) : id.slice(0, 7));
}

function table(rows) {
  const t = el("table");
  for (const r of rows) t.append(el("tr", {}, ...r.map((c, i) => el("td", { class: i === 0 ? "k" : (i >= 2 ? "n" : "") }, c))));
  return el("div", { class: "table-card" }, t);
}

const cardList = () => el("ul", { class: "card" });

function locationSection(d) {
  const l = d.location;
  const out = [el("h3", {}, "Местоположение")];
  if (!l || (l.level === "unknown" && !l.logical)) {
    out.push(el("div", { class: "card sub" }, "Неизвестно"));
    return out;
  }
  const p = el("p", {}, el("span", { class: `badge lvl-${l.level}` }, l.level), " ");
  if (l.point) {
    p.append(`±${meters(l.radius_m)} · ${pct(l.confidence)}`,
      el("br"), el("span", { class: "sub" }, `${l.point.lat.toFixed(6)}, ${l.point.lon.toFixed(6)}`));
    if (l.derived) p.append(el("br"), el("span", { class: "sub" }, "Выведено по связям, здесь никто не наблюдал"));
    if (l.conflict) p.append(el("br"), el("span", { class: "warn" }, "Источники противоречат друг другу"));
  }
  if (l.region) p.append(el("br"), `регион: ${l.region}`);
  if (l.logical) {
    p.append(el("br"), `место: ${l.logical}`);
    if (l.logical_from) p.append(" (от ", devLink(l.logical_from), " по кабелю)");
  }
  out.push(el("div", { class: "card" }, p));
  if (l.sources && l.sources.length) {
    out.push(el("h3", {}, "Источники положения"));
    const ul = cardList();
    for (const s of l.sources) {
      const li = el("li");
      if (s.kind === "link") li.append("от ", devLink(s.from), ` через ${(s.via || []).join("→")}: ±${meters(s.radius_m)}, ${pct(s.confidence)}`);
      else li.append(`${s.method} (${s.observer}): ±${meters(s.radius_m)}, ${pct(s.confidence)}, ${ago(s.time)}`);
      ul.append(li);
    }
    out.push(ul);
  }
  return out;
}

function classSection(d) {
  const c = d.class;
  if (!c) return [el("h3", {}, "Тип"), el("div", { class: "card sub" }, "неизвестен: нет характерных признаков")];
  const why = cardList();
  for (const r of c.reasons || []) why.append(el("li", {}, r));
  if (c.alt) why.append(el("li", { class: "sub" }, `или ${CLASS_RU[c.alt] || c.alt} · ${pct(c.alt_conf)}`));
  return [el("h3", {}, `Тип: ${CLASS_RU[c.kind] || c.kind} · ${pct(c.conf)}`), why];
}

function showDevice(d) {
  const box = $("details");
  const head = [el("h2", {}, devLabel(d)), el("div", { class: "id" }, d.id)];
  const badges = el("div", { class: "badges" });
  for (const m of d.modes) badges.append(el("span", { class: "badge" }, m));
  if (d.is_observer) badges.append(el("span", { class: "badge" }, "наблюдатель"));
  badges.append(el("span", { class: "badge" }, `${d.observations} набл.`), el("span", { class: "badge" }, ago(d.last)));
  if (d.class) badges.prepend(el("span", { class: `badge type c-${classGroup(d)}` }, `${CLASS_RU[d.class.kind] || d.class.kind} · ${pct(d.class.conf)}`));

  const links = state.graph.links.filter((l) => l.a === d.id || l.b === d.id);
  const linkList = cardList();
  for (const l of links) {
    const other = l.a === d.id ? l.b : l.a;
    linkList.append(el("li", {}, `${l.kind}: `, devLink(other), l.detail ? ` (${l.detail})` : ""));
  }

  const nets = cardList();
  for (const n of d.networks) {
    const nn = byNet.get(n);
    nets.append(el("li", {}, el("a", { class: "dev", onclick: () => select({ type: "network", id: n }) }, nn ? netLabel(nn) : n)));
  }

  const ids = cardList();
  for (const i of d.idents) ids.append(el("li", {}, i));
  const asCard = d.as ? el("div", { class: "card" },
    el("a", { class: "dev", onclick: () => select({ type: "as", id: d.as.asn }) }, `AS${d.as.asn} ${d.as.name}`),
    d.as.country ? el("span", { class: "sub" }, ` · регистрация ${d.as.country}`) : null) : null;
  const addrs = cardList();
  for (const a of d.addrs || []) addrs.append(el("li", {}, `${a.ip} — ${ago(a.last)}`));

  const rows = [];
  for (const k of Object.keys(d.claims).sort()) {
    for (const v of d.claims[k]) rows.push([k, v.value, pct(v.conf / 1000), `×${v.count}`]);
  }
  const ev = cardList();
  for (const e of d.evidence || []) ev.append(el("li", {}, `${e.rule}: ${e.detail}`));

  fill(box,
    ...head, badges,
    ...classSection(d),
    ...locationSection(d),
    asCard ? el("h3", {}, "Интернет") : null, asCard,
    links.length ? el("h3", {}, "Связи") : null, links.length ? linkList : null,
    d.networks.length ? el("h3", {}, "Сети") : null, d.networks.length ? nets : null,
    el("h3", {}, "Идентификаторы"), ids,
    (d.addrs || []).length ? el("h3", {}, "Адреса") : null, (d.addrs || []).length ? addrs : null,
    el("h3", {}, "Утверждения"), table(rows),
    (d.evidence || []).length ? el("h3", {}, "Почему это одно устройство") : null, (d.evidence || []).length ? ev : null,
  );
}

function showNetwork(n) {
  const rows = [];
  for (const k of Object.keys(n.claims).sort()) for (const v of n.claims[k]) rows.push([k, v.value]);
  const list = cardList();
  for (const id of n.devices.slice(0, 300)) list.append(el("li", {}, devLink(id)));
  fill($("details"),
    el("h2", {}, netLabel(n)), el("div", { class: "id" }, n.id),
    n.observer_local ? el("div", { class: "card sub" }, "Идентификатор локален для наблюдателя: шлюз не был виден") : null,
    n.as ? el("h3", {}, "Выход в интернет") : null,
    n.as ? el("div", { class: "card" }, el("a", { class: "dev", onclick: () => select({ type: "as", id: n.as.asn }) }, `AS${n.as.asn} ${n.as.name}`)) : null,
    el("h3", {}, "Свойства"), table(rows),
    el("h3", {}, `Устройства (${n.devices.length})`), list,
  );
}

function showAS(asn) {
  const gr = (state.graph.ases || []).find((g) => g.asn === asn);
  if (!gr) return;
  const devs = cardList();
  for (const id of gr.devices) devs.append(el("li", {}, devLink(id)));
  const nets = cardList();
  for (const id of gr.networks || []) {
    const n = byNet.get(id);
    nets.append(el("li", {}, el("a", { class: "dev", onclick: () => select({ type: "network", id }) }, n ? netLabel(n) : id)));
  }
  fill($("details"),
    el("h2", {}, `AS${gr.asn}`), el("div", { class: "id" }, gr.name),
    el("div", { class: "card sub" }, "Автономная система по справочнику IP→AS. Страна — место регистрации, а не расположения."),
    gr.country ? el("div", { class: "badges" }, el("span", { class: "badge" }, gr.country)) : null,
    (gr.networks || []).length ? el("h3", {}, "Сети выходят в интернет здесь") : null, (gr.networks || []).length ? nets : null,
    gr.devices.length ? el("h3", {}, `Адреса в этой AS (${gr.devices.length})`) : null, gr.devices.length ? devs : null,
  );
}

function select(s) {
  state.selected = s;
  if (!s) {
    $("details").replaceChildren(el("p", { class: "hint" }, "Выберите устройство или сеть"));
    if (cy) cy.elements().unselect();
    return;
  }
  if (s.type === "as") showAS(s.id);
  else if (s.type === "device" && byDev.has(s.id)) showDevice(byDev.get(s.id));
  else if (s.type === "network" && byNet.has(s.id)) showNetwork(byNet.get(s.id));
  if (cy) {
    cy.elements().unselect();
    const n = cy.getElementById(s.type === "device" ? s.id : s.type === "as" ? "as:" + s.id : "n:" + s.id);
    if (n.nonempty()) { n.select(); if (state.view === "graph") cy.animate({ center: { eles: n } }, { duration: 200 }); }
  }
  if (map && s.type === "device") {
    const d = byDev.get(s.id);
    if (d && d.location && d.location.point && state.view === "map") map.panTo([d.location.point.lat, d.location.point.lon]);
  }
}

// ---------- render ----------

function render() {
  if (!state.graph) return;
  const f = filters();
  const devs = state.graph.devices.filter((d) => visible(d, f));
  const located = devs.filter((d) => d.location && d.location.point).length;
  $("summary").textContent =
    `${devs.length} из ${state.graph.devices.length} устройств · ${state.graph.networks.length} сет. · ` +
    `${state.graph.links.length} связ. · ${located} с местом`;
  if (state.view === "graph") renderGraph(devs, f);
  else renderMap(devs, f);
  if (state.selected) select(state.selected);
}

// ---------- time ----------

function sliderToTime(v) {
  const { first, last } = state.meta;
  return Math.round(first + (last - first) * (v / 1000));
}

function updateTimeLabel() {
  const v = Number($("at").value);
  $("at-label").textContent = v >= 1000 ? "Сейчас" : fmtTime(sliderToTime(v));
  $("at-now").classList.toggle("hidden", v >= 1000);
}

let timer = null;
function onTime() {
  updateTimeLabel();
  clearTimeout(timer);
  timer = setTimeout(() => {
    const v = Number($("at").value);
    state.at = v >= 1000 ? 0 : sliderToTime(v);
    loadGraph().catch(fail);
  }, 250);
}

// While looking at "now", pick up what a running `rakkan watch` adds.
async function poll() {
  if (state.at !== 0 || document.hidden) return;
  const before = state.meta.observations;
  await loadMeta();
  if (state.meta.observations !== before) await loadGraph();
}

function fail(e) {
  console.error(e);
  status("ошибка: " + e.message);
}

// ---------- wiring ----------

function setView(v) {
  state.view = v;
  if (location.hash !== (v === "map" ? "#map" : "") && !(v === "graph" && location.hash === "")) {
    history.replaceState(null, "", v === "map" ? "#map" : location.pathname);
  }
  $("tab-graph").classList.toggle("active", v === "graph");
  $("tab-map").classList.toggle("active", v === "map");
  $("tab-graph").setAttribute("aria-selected", v === "graph");
  $("tab-map").setAttribute("aria-selected", v === "map");
  $("graph").classList.toggle("hidden", v !== "graph");
  $("map").classList.toggle("hidden", v !== "map");
  $("map-empty").classList.add("hidden");
  render();
  if (v === "map" && map) map.invalidateSize();
  if (v === "graph" && cy) cy.resize();
}

async function main() {
  $("tab-graph").addEventListener("click", () => setView("graph"));
  $("tab-map").addEventListener("click", () => setView("map"));
  $("at").addEventListener("input", onTime);
  $("at-now").addEventListener("click", () => { $("at").value = 1000; onTime(); });
  let qTimer = null;
  $("q").addEventListener("input", () => { updateSearchHint(); clearTimeout(qTimer); qTimer = setTimeout(render, 150); });
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey) {
      const s = keySuggestions();
      if (s.length) { e.preventDefault(); completeKey(s[0].name); }
    } else if (e.key === "Escape") {
      toggleSearchHelp(false);
    }
  });
  $("q-help-btn").addEventListener("click", () => toggleSearchHelp());
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
    if (e.key === "/" && !typing) { e.preventDefault(); $("q").focus(); }
  });
  for (const id of ["net", "cls", "only-located", "hide-ap"]) $(id).addEventListener("change", render);
  for (const c of document.querySelectorAll(".mode, .kind")) c.addEventListener("change", render);
  try {
    await loadMeta();
    if (!state.meta.observations) status("карта пуста: запустите rakkan scan или rakkan watch");
    await loadGraph();
    if (location.hash === "#map") setView("map");
  } catch (e) { fail(e); }
  window.addEventListener("hashchange", () => setView(location.hash === "#map" ? "map" : "graph"));
  setInterval(() => poll().catch(fail), 30000);
}

main();
