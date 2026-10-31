import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { connect, TaskError } from '../index.js';
import { startCluster } from './cluster.js';

const fixture = (name) => readFileSync(new URL(`fixtures/${name}`, import.meta.url));
const echo = fixture('echo.wasm');
const search = fixture('search.wasm');
const spin = fixture('spin.wasm');

// The hash search.wasm inverts (see wasmtest.Mix32).
function mix32(x) {
  x = Math.imul(x, 0x9e3779b1) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  return (x ^ (x >>> 13)) >>> 0;
}

function searchInput(start, count, target) {
  const b = Buffer.alloc(12);
  b.writeUInt32LE(start, 0);
  b.writeUInt32LE(count, 4);
  b.writeUInt32LE(target, 8);
  return b;
}

let cluster;
let mesh;

before(async () => {
  cluster = await startCluster({ nodes: 3, workers: 2 });
  mesh = connect({ socket: cluster.sockets[0] });
});

after(() => cluster?.stop());

test('status shows the network', async () => {
  const st = await mesh.status();
  assert.match(st.id, /^12D3KooW/);
  assert.equal(st.members.filter((m) => m.connected).length, 2);
  assert.equal(st.load.slots, 2);
});

test('run returns output, gas and the executing node', async () => {
  const res = await mesh.run({ module: echo, input: 'hello mesh' });
  assert.equal(res.exitCode, 0);
  assert.equal(res.output.toString(), 'hello mesh');
  assert.ok(res.gasUsed > 0);
  assert.match(res.node, /^12D3KooW/);
  assert.equal(res.attempts, 1);
});

test('json input', async () => {
  const res = await mesh.run({ module: echo, json: { n: 42 } });
  assert.deepEqual(JSON.parse(res.output), { n: 42 });
});

test('task errors reject with their kind', async () => {
  await assert.rejects(mesh.run({ module: spin, limits: { gas: 10_000 } }), (err) => {
    assert.ok(err instanceof TaskError);
    assert.equal(err.kind, 'out_of_gas');
    return true;
  });
});

test('replicas agree on output and gas', async () => {
  const res = await mesh.run({ module: echo, input: 'verify me', replicas: 3 });
  assert.equal(res.replicas.length, 3);
  assert.ok(res.replicas.every((r) => r.agreed));
  assert.equal(new Set(res.replicas.map((r) => r.node)).size, 3, 'replicas ran on distinct nodes');
});

test('map finds a hash preimage across the network and stops', async () => {
  const secret = Math.floor(Math.random() * (1 << 22));
  const target = mix32(secret);
  const chunk = 1 << 16;
  const inputs = [];
  for (let start = 0; start < 1 << 22; start += chunk) inputs.push(searchInput(start, chunk, target));

  let found = null;
  let reported = 0;
  const nodes = new Set();
  for await (const item of mesh.map({ module: search, inputs, stopOnExitCode: 0 })) {
    reported++;
    assert.equal(item.error, undefined);
    nodes.add(item.node);
    if (item.exitCode === 0) found = item.output.readUInt32LE(0);
  }
  assert.equal(found, secret);
  assert.ok(reported <= inputs.length);
  assert.ok(nodes.size > 1, `work spread over ${nodes.size} node(s)`);
});

test('limits above every node\'s policy are explained', async () => {
  await assert.rejects(mesh.run({ module: echo, limits: { timeoutMs: 3_600_000 } }), (err) => {
    assert.equal(err.kind, 'no_capacity');
    assert.match(err.message, /exceed every executor's policy.*timeout/);
    return true;
  });
});

test('breaking out of a map cancels the rest', async () => {
  // 30 tasks spinning ~2s each would keep the 6 slots busy for ~10s.
  const inputs = Array.from({ length: 30 }, () => Buffer.alloc(0));
  for await (const item of mesh.map({ module: spin, inputs, limits: { gas: 2_000_000_000 } })) {
    assert.equal(item.error?.kind, 'out_of_gas');
    break;
  }
  const started = Date.now();
  for (;;) {
    const st = await mesh.status();
    const busy = st.load.busy + st.load.queued + st.members.reduce((n, m) => n + (m.load ? m.load.busy + m.load.queued : 0), 0);
    if (busy === 0) break;
    assert.ok(Date.now() - started < 4_000, `still ${busy} tasks running after the map was abandoned`);
    await new Promise((r) => setTimeout(r, 100));
  }
});

test('no node on the socket', async () => {
  const nowhere = connect({ socket: '/nonexistent/rakkan.sock' });
  await assert.rejects(nowhere.status(), (err) => err instanceof TaskError && err.kind === 'unavailable');
});
