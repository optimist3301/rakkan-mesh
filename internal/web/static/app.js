"use strict";

// --- text --------------------------------------------------------------

const T = {
  ru: {
    tabNet: "Сеть", tabChat: "Чат", tabModels: "Модели", archLlama: "llama-подобная",
    importTitle: "Загрузить модель с Hugging Face",
    importText: "rakkan-web скачает файлы модели и передаст их узлу; узлы с -llm потом возьмут себе её слои. Подходят Llama 3, Qwen2/2.5 и Qwen3 в safetensors, а также DeepSeek-V4.1.",
    repo: "Репозиторий", revision: "Версия", quant: "Хранить веса", modelName: "Имя", namePh: "по имени репозитория",
    q8: "q8_0 — вдвое меньше, почти без потерь", q4m: "q4_m — втрое меньше", q40: "q4_0 — втрое меньше, грубее",
    bf16: "bf16 — как есть, точно", asIs: "как в файлах",
    hfToken: "Токен Hugging Face (для закрытых моделей, например Llama)", check: "Проверить", download: "Скачать и загрузить в сеть",
    importsTitle: "Загрузки", noJobs: "Пока ничего не загружалось.",
    files: "файлов: {0}, всего {1}", archLine: "Архитектура: {0} · коммит {1}",
    noteDeepseek: "DeepSeek-V4.1: rakkan-web ничего не скачивает — узлы с -llm возьмут свои слои с Hub сами (весов {0}). Для закрытого репозитория токен нужен самому узлу (HF_TOKEN).",
    noteMistral: "Mistral пока не поддерживается: его токенизатор (SentencePiece с byte fallback) узел не умеет.",
    noteUnsupported: "Архитектура {0} не поддерживается: подходят llama, qwen2, qwen3 и DeepSeek-V4.1.",
    st: { downloading: "скачивается", importing: "узел режет модель на слои", done: "готово", failed: "ошибка", cancelled: "отменено" },
    cancel: "Отменить", openChat: "Открыть в чате", eta: "осталось ~{0}", speedOf: "{0}/с",
    modelReady: "ID модели: {0}", thisNode: "Этот узел", network: "Сеть", version: "Версия",
    addresses: "Адреса", connected: "Подключено", guests: "Гости", llmNodes: "Узлы нейросетей",
    ownLoad: "Своя нагрузка", members: "Участники", showOffline: "показывать отключённых",
    node: "Узел", role: "Роль", state: "Состояние", path: "Путь", load: "Нагрузка", caps: "Возможности",
    noMembers: "Пока никого не видно. Узел ищет сеть в DHT и через Yggdrasil — это может занять минуту.",
    llmTitle: "Нейросети в сети", device: "Устройство", speed: "Скорость", freeMem: "Свободно",
    shards: "Держит слои", noLLM: "Ни один узел не обслуживает нейросети. Запустите узел с флагом -llm.",
    model: "Модель", modelIdPh: "ID модели", add: "Добавить", temperature: "Температура",
    maxTokens: "Длина ответа, токенов", draft: "черновая модель (быстрее)", newChat: "Новый разговор",
    chatHint: "Выберите модель и напишите сообщение. Ответ считают узлы сети: каждый — свой кусок слоёв.",
    image: "Картинка", messagePh: "Сообщение…", send: "Отправить", stop: "Стоп",
    tokenTitle: "Нужен токен", tokenText: "Узел слушает не только localhost и просит токен из флага -web-token.", ok: "Войти",
    online: "узел на связи", offline: "нет связи с узлом", of: "из", known: "известно",
    open: "сеть открыта, PoW {0} бит", closed: "сеть закрыта для гостей", youGuest: "этот узел — гость",
    peerings: "пиров Yggdrasil", yggOff: "выключен", yggNoPeers: "нет пиров — через Yggdrasil не дозвониться",
    slots: "слотов занято", queued: "в очереди: {0}",
    on: "на связи", off: "отключён", direct: "напрямую", relayed: "через ретранслятор", ygg: "через Yggdrasil",
    member: "участник", admin: "админ", guest: "гость", score: "репутация", seen: "был {0}",
    cpu: "процессор", noModels: "нет моделей", noModelsHint: "Пока ни один узел не держит слои моделей. Вставьте ID модели (его возвращает импорт модели через SDK или API) и нажмите «Добавить».",
    noChat: "у модели нет формата чата: она просто продолжит текст", layers: "слоёв: {0}",
    images: "понимает картинки", copied: "скопировано",
    stats: "{0} ток. · {1} ток/с · этапов: {2}", failovers: " · переключений: {0}",
    errNoCapacity: "Сейчас никто не может посчитать эту модель: нет узлов с -llm или им не хватает памяти.",
    errLLM: "На этом узле нейросети не включены.", stopped: "остановлено", ago: { s: "{0} с назад", m: "{0} мин назад", h: "{0} ч назад", d: "{0} дн назад" },
    badModel: "Не похоже на ID модели", notFound: "Модель не найдена: {0}",
    node_failed: "Узел оборвал запрос. Подробности — в журнале rakkan-node.",
    node_unreachable: "rakkan-web не достучался до узла. Запущен ли rakkan-node?",
    oldNode: "Узел старше 0.007-alpha: модель не видит прошлых реплик разговора, а имена моделей неизвестны. Обновите rakkan-node.",
    emptyAnswer: "(пустой ответ: модель сразу закончила)", lost: "Соединение с узлом прервалось. Загляните в журнал узла.",
  },
  en: {
    tabNet: "Network", tabChat: "Chat", tabModels: "Models", archLlama: "llama-like",
    importTitle: "Load a model from Hugging Face",
    importText: "rakkan-web downloads the model's files and hands them to the node; nodes with -llm then take its layers. Llama 3, Qwen2/2.5 and Qwen3 in safetensors fit, and DeepSeek-V4.1.",
    repo: "Repository", revision: "Revision", quant: "Store weights as", modelName: "Name", namePh: "the repository's name",
    q8: "q8_0 — half the size, nearly lossless", q4m: "q4_m — a third of the size", q40: "q4_0 — a third, coarser",
    bf16: "bf16 — as is, exact", asIs: "as in the files",
    hfToken: "Hugging Face token (for gated models such as Llama)", check: "Check", download: "Download into the network",
    importsTitle: "Imports", noJobs: "Nothing imported yet.",
    files: "{0} files, {1} in all", archLine: "Architecture: {0} · commit {1}",
    noteDeepseek: "DeepSeek-V4.1: rakkan-web downloads nothing — the -llm nodes fetch their layers from the Hub themselves ({0} of weights). A gated repository needs the token on the node (HF_TOKEN).",
    noteMistral: "Mistral is not supported yet: the node cannot read its tokenizer (SentencePiece with byte fallback).",
    noteUnsupported: "Architecture {0} is not supported: llama, qwen2, qwen3 and DeepSeek-V4.1 are.",
    st: { downloading: "downloading", importing: "the node is splitting it into layers", done: "done", failed: "failed", cancelled: "cancelled" },
    cancel: "Cancel", openChat: "Open in chat", eta: "~{0} left", speedOf: "{0}/s",
    modelReady: "Model ID: {0}", thisNode: "This node", network: "Network", version: "Version",
    addresses: "Addresses", connected: "Connected", guests: "Guests", llmNodes: "Model nodes",
    ownLoad: "Own load", members: "Members", showOffline: "show disconnected",
    node: "Node", role: "Role", state: "State", path: "Path", load: "Load", caps: "Capabilities",
    noMembers: "Nobody in sight yet. The node is looking for the network in the DHT and over Yggdrasil — this can take a minute.",
    llmTitle: "Language models", device: "Device", speed: "Speed", freeMem: "Free",
    shards: "Holds layers", noLLM: "No node serves language models. Start a node with -llm.",
    model: "Model", modelIdPh: "model ID", add: "Add", temperature: "Temperature",
    maxTokens: "Answer length, tokens", draft: "draft model (faster)", newChat: "New conversation",
    chatHint: "Choose a model and write a message. Nodes of the network compute the answer, each its share of the layers.",
    image: "Image", messagePh: "Message…", send: "Send", stop: "Stop",
    tokenTitle: "Token needed", tokenText: "The node listens beyond localhost and asks for the token of -web-token.", ok: "Sign in",
    online: "node online", offline: "node unreachable", of: "of", known: "known",
    open: "open network, PoW {0} bits", closed: "closed to guests", youGuest: "this node is a guest",
    peerings: "Yggdrasil peers", yggOff: "off", yggNoPeers: "no peers — unreachable over Yggdrasil",
    slots: "slots busy", queued: "queued: {0}",
    on: "online", off: "offline", direct: "direct", relayed: "relayed", ygg: "Yggdrasil",
    member: "member", admin: "admin", guest: "guest", score: "score", seen: "seen {0}",
    cpu: "CPU", noModels: "no models", noModelsHint: "No node holds a model's layers yet. Paste a model ID (importing a model through the SDK or the API returns it) and press Add.",
    noChat: "the model has no chat format: it will just continue the text", layers: "{0} layers",
    images: "takes images", copied: "copied",
    stats: "{0} tokens · {1} tok/s · {2} stages", failovers: " · {0} failovers",
    errNoCapacity: "Nobody can run this model now: no -llm nodes, or not enough memory.",
    errLLM: "Language models are not enabled on this node.", stopped: "stopped", ago: { s: "{0}s ago", m: "{0} min ago", h: "{0} h ago", d: "{0} d ago" },
    badModel: "That does not look like a model ID", notFound: "Model not found: {0}",
    node_failed: "The node broke off the request. See rakkan-node's log.",
    node_unreachable: "rakkan-web cannot reach the node. Is rakkan-node running?",
    oldNode: "The node is older than 0.007-alpha: the model does not see the conversation's earlier turns, and model names are unknown. Update rakkan-node.",
    emptyAnswer: "(empty answer: the model stopped at once)", lost: "The connection to the node broke. See the node's log.",
  },
};
const lang = (navigator.language || "ru").toLowerCase().startsWith("ru") ? "ru" : "en";
const L = T[lang];
const fmt = (s, ...a) => s.replace(/\{(\d)\}/g, (_, i) => a[i]);
document.documentElement.lang = lang;
for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = L[el.dataset.i18n];
for (const el of document.querySelectorAll("[data-i18n-ph]")) el.placeholder = L[el.dataset.i18nPh];
for (const el of document.querySelectorAll("[data-i18n-title]")) el.title = L[el.dataset.i18nTitle];

// --- helpers -----------------------------------------------------------

const $ = (id) => document.getElementById(id);
const hint = $("chat-hint"); // detached while a conversation shows
let attachments = []; // images to send, as data: URLs
const store = {
  get(k, d) { try { const v = localStorage.getItem("rakkan." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("rakkan." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
const short = (id) => id.length > 16 ? id.slice(0, 8) + "…" + id.slice(-6) : id;
const bytes = (n) => {
  if (!n) return "0";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (n >= 10 || i === 0 ? n.toFixed(0) : n.toFixed(1)) + " " + u[i];
};
function ago(t) {
  const s = Math.max(0, (Date.now() - new Date(t).getTime()) / 1000);
  if (s < 60) return fmt(L.ago.s, Math.round(s));
  if (s < 3600) return fmt(L.ago.m, Math.round(s / 60));
  if (s < 86400) return fmt(L.ago.h, Math.round(s / 3600));
  return fmt(L.ago.d, Math.round(s / 86400));
}
async function copy(text, node) {
  try { await navigator.clipboard.writeText(text); } catch { return; }
  const old = node.title;
  node.title = L.copied;
  setTimeout(() => { node.title = old; }, 1200);
}
function peerButton(id) {
  return el("button", { class: "peer", title: id, onclick: (e) => copy(id, e.currentTarget) }, el("span", { class: "mono", text: short(id) }));
}

// --- API ---------------------------------------------------------------

let token = store.get("token", "");
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers);
  if (token) headers.Authorization = "Bearer " + token;
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    await askToken();
    return api(path, opts);
  }
  return res;
}
let tokenWait = null;
function askToken() {
  if (tokenWait) return tokenWait;
  tokenWait = new Promise((resolve) => {
    const d = $("token-dialog");
    d.addEventListener("close", () => {
      token = $("token-input").value.trim();
      store.set("token", token);
      tokenWait = null;
      resolve();
    }, { once: true });
    d.showModal();
  });
  return tokenWait;
}
async function errorText(res) {
  try {
    const j = await res.json();
    if (j.error && L[j.error.kind]) return L[j.error.kind];
    if (j.error) return typeof j.error === "string" ? j.error : j.error.message;
  } catch { /* not JSON */ }
  return res.status + " " + res.statusText;
}

// --- tabs --------------------------------------------------------------

const TABS = ["net", "chat", "models"];
function showTab(name) {
  if (!TABS.includes(name)) name = "net";
  for (const t of TABS) {
    $("tab-" + t).setAttribute("aria-selected", String(t === name));
    $("view-" + t).hidden = t !== name;
  }
  store.set("tab", name);
  history.replaceState(null, "", "#" + name);
  if (name === "chat") $("input").focus();
  if (name === "models") pollJobs();
}
for (const t of TABS) $("tab-" + t).addEventListener("click", () => showTab(t));

// --- monitoring --------------------------------------------------------

let status = null;

function pathOf(m) {
  if (!m.connected) return null;
  const a = m.addr || "";
  if (a.includes("/ygg/")) return ["ygg", L.ygg];
  if (m.relayed || a.includes("/p2p-circuit")) return ["relayed", L.relayed];
  let t = "";
  if (a.includes("/quic")) t = " · QUIC";
  else if (a.includes("/tcp/")) t = " · TCP";
  return ["direct", L.direct + t];
}

function loadCell(l) {
  if (!l || !l.slots) return el("span", { class: "caps", text: "—" });
  const bar = el("span", { class: "bar" }, el("i"));
  bar.firstChild.style.width = Math.min(100, (100 * l.busy) / l.slots) + "%";
  const q = l.queued ? " +" + l.queued : "";
  return el("span", { class: "state" }, bar, el("span", { class: "num", text: l.busy + "/" + l.slots + q }));
}

// A node older than 0.007-alpha reports no addresses; it ignores a
// conversation's history and cannot describe models.
let oldNode = false;

function renderStatus(s) {
  status = s;
  if (oldNode !== !("addrs" in s)) { oldNode = !oldNode; onModelChange(); }
  $("self-id").textContent = s.id;
  $("self-net").textContent = s.network;
  $("self-version").textContent = s.version || "—";
  const addrs = $("self-addrs");
  addrs.replaceChildren(...(s.addrs || []).map((a) => el("li", { text: a + "/p2p/" + s.id })));

  const members = s.members || [];
  const on = members.filter((m) => m.connected);
  $("st-connected").textContent = on.length;
  $("st-known").textContent = L.known + ": " + members.length;
  $("st-guests").textContent = members.filter((m) => m.guest && m.connected).length;
  $("st-open").textContent = s.guest ? L.youGuest : s.open ? fmt(L.open, s.pow_bits || 0) : L.closed;
  const llm = s.llm || [];
  $("st-llm").textContent = llm.length;
  $("st-llm-sub").textContent = llm.length ? bytes(llm.reduce((a, n) => a + (n.free_mem || 0), 0)) + " " + L.freeMem.toLowerCase() : "";
  if (s.yggdrasil) {
    $("st-ygg").textContent = s.yggdrasil.peers;
    $("st-ygg-sub").textContent = s.yggdrasil.peers ? L.peerings : L.yggNoPeers;
  } else {
    $("st-ygg").textContent = "—";
    $("st-ygg-sub").textContent = L.yggOff;
  }
  if (s.yggdrasil) addrs.append(el("li", { text: s.yggdrasil.addr + "/p2p/" + s.id }));
  const l = s.load || {};
  $("st-load").textContent = (l.busy || 0) + "/" + (l.slots || 0);
  $("st-queue").textContent = l.queued ? fmt(L.queued, l.queued) : L.slots;

  // Members: connected first, then by RTT, then by ID.
  const showOff = $("show-offline").checked;
  const rows = members
    .filter((m) => showOff || m.connected)
    .sort((a, b) => (b.connected - a.connected) || (a.rtt_ms - b.rtt_ms) || a.id.localeCompare(b.id))
    .map((m) => {
      const roles = m.guest ? [L.guest] : (m.roles || []).map((r) => L[r] || r);
      const roleCell = el("td", {}, ...roles.map((r) => el("span", { class: "badge" + (m.guest ? " muted" : ""), text: r })));
      if (m.guest && m.score != null) roleCell.append(el("span", { class: "caps", text: L.score + " " + m.score.toFixed(2) }));
      const p = pathOf(m);
      const state = el("span", { class: "state" }, el("span", { class: "dot " + (m.connected ? "ok" : "") }),
        el("span", { text: m.connected ? L.on : m.last_seen ? fmt(L.seen, ago(m.last_seen)) : L.off }));
      return el("tr", { class: m.connected ? "" : "offline" },
        el("td", {}, peerButton(m.id)),
        roleCell,
        el("td", {}, state),
        el("td", { title: m.addr || "" }, p ? el("span", { class: "badge" + (p[0] === "direct" ? "" : " muted"), text: p[1] }) : "—"),
        el("td", { class: "num", text: m.connected && m.rtt_ms ? m.rtt_ms.toFixed(m.rtt_ms < 10 ? 1 : 0) + " ms" : "—" }),
        el("td", {}, loadCell(m.load)),
        el("td", { class: "caps", text: (m.load && m.load.capabilities || []).join(", ") || "—" }));
    });
  $("members").tBodies[0].replaceChildren(...rows);
  $("members-empty").hidden = rows.length > 0;

  $("llm").tBodies[0].replaceChildren(...llm.map((n) => el("tr", {},
    el("td", {}, peerButton(n.id)),
    el("td", { text: n.device || L.cpu }),
    el("td", { class: "num", text: (n.speed || 0).toFixed(1) + " GB/s" }),
    el("td", { class: "num", text: bytes(n.free_mem) + (n.free_cache ? " + " + bytes(n.free_cache) : "") }),
    el("td", {}, ...(n.shards || []).map((sh) => el("span", { class: "badge muted", title: sh.model, text: modelLabel(sh.model) + " " + sh.from + "–" + sh.to })))
  )));
  $("llm-empty").hidden = llm.length > 0;

  updateModels();
}

async function poll() {
  try {
    const res = await api("/v1/status");
    if (!res.ok) throw new Error(await errorText(res));
    renderStatus(await res.json());
    setConn(true);
  } catch {
    setConn(false);
  }
}
function setConn(ok) {
  $("conn").firstElementChild.className = "dot " + (ok ? "ok" : "bad");
  $("conn-text").textContent = ok ? L.online : L.offline;
}
$("show-offline").checked = store.get("showOffline", true);
$("show-offline").addEventListener("change", (e) => { store.set("showOffline", e.target.checked); if (status) renderStatus(status); });

let timer = null;
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(async () => { await poll(); schedule(); }, document.hidden ? 15000 : 3000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) { poll(); schedule(); } });
poll().then(schedule);

// --- models ------------------------------------------------------------

const models = new Map(); // id -> info, or null while unknown
const failed = new Set();
const describing = new Set();
const savedModels = store.get("models", []);
for (const id of savedModels) models.set(id, null);

function modelLabel(id) {
  const info = models.get(id);
  return info && info.name ? info.name : short(id);
}

async function describe(id) {
  if (failed.has(id)) return;
  try {
    const res = await api("/v1/llm/models/" + encodeURIComponent(id));
    if (!res.ok) {
      failed.add(id);
      return;
    }
    models.set(id, await res.json());
    renderModelSelect();
    if (status) renderStatus(status);
  } catch { /* retried with the next status */ }
}

function updateModels() {
  let added = false;
  for (const n of (status && status.llm) || []) {
    for (const sh of n.shards || []) {
      if (!models.has(sh.model)) { models.set(sh.model, null); added = true; }
    }
  }
  for (const [id, info] of models) if (info === null && !describing.has(id)) { describing.add(id); describe(id).finally(() => describing.delete(id)); }
  if (added) renderModelSelect();
}
function renderModelSelect() {
  const sel = $("model");
  const current = sel.value || store.get("model", "");
  const ids = [...models.keys()];
  if (!ids.length) {
    sel.replaceChildren(el("option", { value: "", text: L.noModels }));
    sel.disabled = true;
  } else {
    sel.disabled = false;
    sel.replaceChildren(...ids.map((id) => el("option", { value: id, text: modelLabel(id) })));
    if (ids.includes(current)) sel.value = current;
  }
  onModelChange();
}

function currentModel() {
  const id = $("model").value;
  return id ? { id, info: models.get(id) } : null;
}

function onModelChange() {
  const m = currentModel();
  store.set("model", m ? m.id : "");
  const info = m && m.info;
  const parts = [];
  if (info) {
    parts.push(info.arch === "llama" ? L.archLlama : info.arch);
    if (info.layers) parts.push(fmt(L.layers, info.layers));
    if (info.images) parts.push(L.images);
    if (!info.chat) parts.push(L.noChat);
  }
  $("model-info").textContent = m ? parts.join(" · ") : "";
  $("model-info").title = m ? m.id : "";
  $("draft-row").hidden = !(info && info.draft);
  $("attach-label").hidden = !(info && info.images);
  if (!(info && info.images)) { attachments = []; renderAttachments(); }
  hint.textContent = (m ? L.chatHint : L.noModelsHint) + (oldNode ? "\n\n" + L.oldNode : "");
}
$("model").addEventListener("change", onModelChange);

$("add-model").addEventListener("submit", (e) => {
  e.preventDefault();
  const id = $("add-model-id").value.trim();
  if (!/^[0-9a-f]{64}$/i.test(id) && !/^[A-Za-z0-9]{40,}$/.test(id)) {
    $("model-info").textContent = L.badModel;
    return;
  }
  $("add-model-id").value = "";
  failed.delete(id);
  if (!models.has(id)) models.set(id, null);
  const saved = store.get("models", []);
  if (!saved.includes(id)) { saved.push(id); store.set("models", saved); }
  store.set("model", id);
  renderModelSelect();
  $("model").value = id;
  onModelChange();
  describe(id).then(() => { if (failed.has(id)) $("model-info").textContent = fmt(L.notFound, short(id)); });
});
renderModelSelect();

// --- chat --------------------------------------------------------------

const temp = $("temp");
temp.value = store.get("temperature", 0.7);
$("temp-out").textContent = temp.value;
temp.addEventListener("input", () => { $("temp-out").textContent = temp.value; store.set("temperature", Number(temp.value)); });
$("max-tokens").value = store.get("maxTokens", 512);
$("max-tokens").addEventListener("change", (e) => store.set("maxTokens", Number(e.target.value)));
$("draft").checked = store.get("draft", true);
$("draft").addEventListener("change", (e) => store.set("draft", e.target.checked));

// A conversation: [{user, assistant, images?, stats?, error?}]. Images are
// not kept across reloads: they may be large.
let convo = store.get("convo", []).filter((t) => t.assistant || t.error || t.stats);
let busy = null; // AbortController of the running generation

function saveConvo() {
  store.set("convo", convo.map(({ user, assistant, stats, error, model }) => ({ user, assistant, stats, error, model })));
}

// renderText shows a message as text, with ``` fences as code blocks.
function renderText(node, text, done) {
  node.replaceChildren();
  if (!text && done) { node.append(el("span", { class: "caps", text: L.emptyAnswer })); return; }
  const parts = text.split("```");
  parts.forEach((p, i) => {
    if (i % 2 === 0) { if (p) node.append(document.createTextNode(p)); return; }
    const nl = p.indexOf("\n");
    node.append(el("pre", {}, el("code", { text: nl >= 0 ? p.slice(nl + 1) : p })));
  });
}

function messageNodes(turn) {
  const out = [];
  const u = el("div", { class: "msg user" });
  if (turn.images && turn.images.length) u.append(el("div", { class: "imgs" }, ...turn.images.map((src) => el("img", { src, alt: "" }))));
  u.append(document.createTextNode(turn.user));
  out.push(u);
  if (turn.error) {
    out.push(el("div", { class: "msg error", text: turn.error }));
  } else {
    const b = el("div", { class: "msg bot" });
    const body = el("div");
    renderText(body, turn.assistant || "", !!turn.stats);
    b.append(body);
    if (turn.stats) b.append(el("div", { class: "meta", text: statsText(turn.stats) }));
    out.push(b);
    turn.node = body;
    turn.bubble = b;
  }
  return out;
}

function statsText(st) {
  let s = fmt(L.stats, st.tokens, (st.tokens_per_s || 0).toFixed(1), (st.stages || []).length);
  if (st.failovers) s += fmt(L.failovers, st.failovers);
  if (st.stopped) s += " · " + L.stopped;
  return s;
}

function renderConvo() {
  const box = $("messages");
  box.replaceChildren();
  if (!convo.length) { box.append(hint); return; }
  for (const t of convo) box.append(...messageNodes(t));
  box.scrollTop = box.scrollHeight;
}

function renderAttachments() {
  $("attachments").replaceChildren(...attachments.map((src, i) => el("figure", {},
    el("img", { src, alt: "" }),
    el("button", { type: "button", class: "danger", text: "×", onclick: () => { attachments.splice(i, 1); renderAttachments(); } }))));
}
$("attach").addEventListener("change", async (e) => {
  for (const f of e.target.files) {
    if (attachments.length >= 8) break;
    attachments.push(await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(f); }));
  }
  e.target.value = "";
  renderAttachments();
});

const input = $("input");
function autosize() { input.style.height = "auto"; input.style.height = Math.min(200, input.scrollHeight) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
});

$("new-chat").addEventListener("click", () => {
  if (busy) busy.abort();
  convo = [];
  saveConvo();
  renderConvo();
  input.focus();
});
$("stop").addEventListener("click", () => { if (busy) busy.abort(); });

function setBusy(ctrl) {
  busy = ctrl;
  $("send").hidden = !!ctrl;
  $("stop").hidden = !ctrl;
  $("model").disabled = !!ctrl || !models.size;
}

$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  const m = currentModel();
  if (!text || busy || !m) return;
  const chat = !m.info || m.info.chat;
  const history = chat ? convo.filter((t) => !t.error && t.model === m.id).map((t) => ({ user: t.user, assistant: t.assistant })) : [];
  const turn = { user: text, assistant: "", images: attachments, model: m.id };
  convo.push(turn);
  input.value = "";
  autosize();
  attachments = [];
  renderAttachments();
  const box = $("messages");
  if (box.firstChild === hint) box.replaceChildren();
  box.append(...messageNodes(turn));
  turn.bubble.classList.add("cursor");
  box.scrollTop = box.scrollHeight;

  const ctrl = new AbortController();
  setBusy(ctrl);
  const req = {
    model: m.id, prompt: text, chat, history,
    max_tokens: Number($("max-tokens").value) || 512,
    temperature: Number(temp.value),
  };
  if (m.info && m.info.draft && $("draft").checked) req.draft = true;
  if (turn.images.length) req.images = turn.images;
  // A model not described (an old node) is asked in the chat format first,
  // and again without it if it has none.
  const ask = async () => {
    try {
      return await stream(req, ctrl, turn, box);
    } catch (err) {
      if (!req.chat || m.info || turn.assistant || !String(err.message).includes("no chat format")) throw err;
      models.set(m.id, { model: m.id, arch: "?", chat: false });
      onModelChange();
      req.chat = false;
      req.history = [];
      return stream(req, ctrl, turn, box);
    }
  };
  try {
    await ask();
  } catch (err) {
    if (err.name === "AbortError") {
      turn.stats = Object.assign({ tokens: 0, stages: [] }, turn.stats, { stopped: true });
    } else {
      let msg = String(err.message || err);
      if (msg.includes("not enabled")) msg = L.errLLM;
      if (err instanceof TypeError) msg = L.lost; // fetch failed or the stream broke
      if (!turn.assistant) {
        turn.error = msg;
      } else {
        turn.stats = Object.assign({ tokens: 0, stages: [] }, turn.stats, { stopped: true });
        turn.bubble.after(el("div", { class: "msg error", text: msg }));
      }
    }
  } finally {
    setBusy(null);
    if (turn.error) {
      turn.bubble.replaceWith(el("div", { class: "msg error", text: turn.error }));
    } else {
      turn.bubble.classList.remove("cursor");
      if (!turn.assistant) renderText(turn.node, "", true);
      if (turn.stats) turn.bubble.append(el("div", { class: "meta", text: statsText(turn.stats) }));
    }
    saveConvo();
    box.scrollTop = box.scrollHeight;
    input.focus();
  }
});

// stream runs a generation, adding its text to turn as it comes.
async function stream(req, ctrl, turn, box) {
  const res = await api("/v1/llm/generate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req), signal: ctrl.signal,
  });
  if (!res.ok) throw new Error(await errorText(res));
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.error) {
        throw new Error(ev.error.kind === "no_capacity" ? L.errNoCapacity : ev.error.message);
      } else if (ev.done) {
        turn.stats = ev.stats;
      } else if (ev.text != null) {
        const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
        turn.assistant += ev.text;
        renderText(turn.node, turn.assistant);
        if (stick) box.scrollTop = box.scrollHeight;
      }
    }
  }
}

renderConvo();

// --- models ------------------------------------------------------------

function duration(s) {
  if (!isFinite(s) || s < 0) return "";
  if (s < 90) return Math.round(s) + (lang === "ru" ? " с" : " s");
  if (s < 5400) return Math.round(s / 60) + (lang === "ru" ? " мин" : " min");
  return (s / 3600).toFixed(1) + (lang === "ru" ? " ч" : " h");
}

function importRequest() {
  return {
    repo: $("imp-repo").value.trim(), revision: $("imp-rev").value.trim() || "main",
    quant: $("imp-quant").value, name: $("imp-name").value.trim(), token: $("imp-token").value.trim(),
  };
}

async function appPost(path, body) {
  const res = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

function showInfo(content, bad) {
  const box = $("imp-info");
  box.hidden = false;
  box.className = "imp-info" + (bad ? " bad" : "");
  box.replaceChildren(...[].concat(content));
}

function describeRepo(info) {
  const out = [el("div", { text: fmt(L.archLine, info.model_type || "?", info.revision.slice(0, 10)) })];
  if (info.deepseek) {
    out.push(el("div", { class: "warn", text: fmt(L.noteDeepseek, bytes(info.total)) }));
  } else {
    out.push(el("div", { text: fmt(L.files, info.files.length, bytes(info.total)) }));
    out.push(el("ul", {}, ...info.files.map((f) => el("li", { class: "mono", text: f.name + " — " + bytes(f.size) }))));
  }
  if (info.note === "mistral") out.push(el("div", { class: "warn", text: L.noteMistral }));
  if (info.note === "unsupported") out.push(el("div", { class: "warn", text: fmt(L.noteUnsupported, info.model_type || "?") }));
  return out;
}

$("imp-repo").addEventListener("input", () => { $("imp-info").hidden = true; });
$("imp-check").addEventListener("click", async () => {
  const req = importRequest();
  if (!req.repo) return $("imp-repo").focus();
  $("imp-check").disabled = true;
  try {
    const info = await appPost("/app/models/inspect", req);
    showInfo(describeRepo(info), !info.supported);
    $("imp-quant").disabled = !!info.deepseek;
  } catch (err) {
    showInfo(el("span", { text: err.message }), true);
  } finally {
    $("imp-check").disabled = false;
  }
});
$("import-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const req = importRequest();
  if ($("imp-quant").disabled) req.quant = "";
  $("imp-start").disabled = true;
  try {
    await appPost("/app/models/import", req);
    $("imp-info").hidden = true;
    pollJobs();
  } catch (err) {
    showInfo(el("span", { text: err.message }), true);
  } finally {
    $("imp-start").disabled = false;
  }
});

// Speeds: the bytes a job had at the last poll.
const lastDone = new Map();
let jobsTimer = null;
const announced = new Set(store.get("importedJobs", []));

function renderJob(j) {
  const now = Date.now();
  const prev = lastDone.get(j.id);
  lastDone.set(j.id, { done: j.done, at: now });
  const running = j.state === "downloading" || j.state === "importing";
  let sub = L.st[j.state] || j.state;
  if (j.state === "downloading" && j.total) {
    sub += " · " + bytes(j.done) + " / " + bytes(j.total);
    if (prev && now > prev.at && j.done > prev.done) {
      const speed = ((j.done - prev.done) * 1000) / (now - prev.at);
      sub += " · " + fmt(L.speedOf, bytes(speed)) + " · " + fmt(L.eta, duration((j.total - j.done) / speed));
    }
  }
  if (j.quant) sub += " · " + j.quant;
  const actions = el("div", { class: "actions" });
  if (running) actions.append(el("button", { class: "secondary", text: L.cancel, onclick: () => appPost("/app/models/jobs/" + j.id + "/cancel", {}).then(pollJobs) }));
  if (j.state === "done" && j.model) actions.append(el("button", { text: L.openChat, onclick: () => openInChat(j.model) }));
  const row = el("div", { class: "job" },
    el("div", {}, el("div", { class: "job-title", text: j.name + "  ·  " + j.repo }), el("div", { class: "job-sub", text: sub })),
    actions);
  if (running || j.state === "done") {
    const bar = el("div", { class: "progress" }, el("i"));
    const frac = j.state === "done" ? 1 : j.total ? j.done / j.total : 0;
    bar.firstChild.style.width = Math.round(frac * 100) + "%";
    row.append(bar);
  }
  if (j.error) row.append(el("div", { class: "err", text: j.error }));
  if (j.model) row.append(el("div", { class: "job-sub mono", title: j.model, text: fmt(L.modelReady, j.model) }));
  return row;
}

function rememberModel(id) {
  if (!models.has(id)) models.set(id, null);
  const saved = store.get("models", []);
  if (!saved.includes(id)) { saved.push(id); store.set("models", saved); }
  failed.delete(id);
  renderModelSelect();
  updateModels();
}

function openInChat(id) {
  rememberModel(id);
  $("model").value = id;
  onModelChange();
  showTab("chat");
}

async function pollJobs() {
  clearTimeout(jobsTimer);
  let jobs = [];
  try {
    const res = await api("/app/models/jobs");
    if (res.ok) jobs = await res.json();
  } catch { /* shown by the connection dot */ }
  $("jobs").replaceChildren(...jobs.map(renderJob));
  $("jobs-empty").hidden = jobs.length > 0;
  for (const j of jobs) {
    if (j.state === "done" && j.model && !announced.has(j.model)) {
      announced.add(j.model);
      store.set("importedJobs", [...announced]);
      rememberModel(j.model);
    }
  }
  const active = jobs.some((j) => j.state === "downloading" || j.state === "importing");
  if (active || !$("view-models").hidden) jobsTimer = setTimeout(pollJobs, active ? 1000 : 5000);
}

showTab(location.hash.slice(1) || store.get("tab", "net"));
pollJobs();
