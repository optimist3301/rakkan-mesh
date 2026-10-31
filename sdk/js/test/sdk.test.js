import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
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
  cluster = await startCluster({ nodes: 3, workers: 2, llm: true });
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
  // In the last chunk, so most chunks run before the stop and the work has
  // to spread (a secret in the first chunks could finish on one node).
  const secret = (1 << 22) - 1 - Math.floor(Math.random() * (1 << 15));
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

test('blobs stream in and out, with ranges', async () => {
  const data = randomBytes(5 * 1024 * 1024 + 99);
  const file = `${cluster.dir}/upload.bin`;
  writeFileSync(file, data);
  const { id, size } = await mesh.putBlob({ path: file });
  assert.equal(size, data.length);
  assert.deepEqual(await mesh.readBlob(id), data);
  const part = await mesh.readBlob(id, { start: 1_048_570, end: 1_048_590 });
  assert.deepEqual(part, data.subarray(1_048_570, 1_048_591));
});

test('large inputs and outputs travel as blobs', async () => {
  const big = randomBytes(3 * 1024 * 1024);
  const echoBig = fixture('echo-big.wasm');
  const res = await mesh.run({ module: echoBig, input: big, limits: { memoryMB: 16 } });
  assert.ok(res.outputBlob, 'output came back as a blob');
  assert.equal(res.outputSize, big.length);
  assert.deepEqual(res.output, big);

  const kept = await mesh.run({ module: echoBig, input: big, limits: { memoryMB: 16 }, outputAsBlob: true });
  assert.equal(kept.output.length, 0);
  assert.equal(kept.outputBlob, res.outputBlob, 'same content, same blob ID');
});

test('image kernels make thumbnails on the network', async () => {
  const res = await mesh.run({ module: fixture('thumbnail-png.wasm'), input: fixture('sample.png'), limits: { memoryMB: 16 } });
  assert.equal(res.exitCode, 0);
  // PNG IHDR: width and height at bytes 16 and 20.
  assert.equal(res.output.readUInt32BE(16), 256);
  assert.equal(res.output.readUInt32BE(20), 160);
  const st = await mesh.status();
  assert.ok(st.load.capabilities.includes('image/1'));
});

test('attached blobs are read in pieces', async () => {
  const data = randomBytes(2 * 1024 * 1024 + 5);
  const { id } = await mesh.putBlob(data);
  const res = await mesh.run({ module: fixture('blob-echo.wasm'), blobs: [id] });
  assert.deepEqual(res.output, data);
});

// Tasks compiled from AssemblyScript (sdk/as).
const asModule = (name) => readFileSync(new URL(`../../as/build/${name}.wasm`, import.meta.url));

test('AssemblyScript: word count as a map over ranges of a shared blob', async () => {
  const line = 'the mesh runs tasks; meshmesh is not mesh-free.\n';
  const text = Buffer.from(line.repeat(200_000)); // ~9.6 MB
  const { id } = await mesh.putBlob(text);
  const word = 'mesh';
  const step = Math.ceil(text.length / 12);
  const inputs = [];
  for (let off = 0; off < text.length; off += step) {
    const b = Buffer.alloc(16 + word.length);
    b.writeBigUInt64LE(BigInt(off), 0);
    b.writeBigUInt64LE(BigInt(step), 8);
    b.write(word, 16);
    inputs.push(b);
  }
  let total = 0;
  for await (const item of mesh.map({ module: asModule('wordcount'), inputs, blobs: [id] })) {
    assert.equal(item.error, undefined);
    total += item.output.readUInt32LE(0);
  }
  assert.equal(total, 200_000 * 4); // mesh, meshmesh (2), mesh-free
});

test('AssemblyScript: sepia photo through image kernels and pixel access', async () => {
  const res = await mesh.run({ module: asModule('sepia'), input: fixture('sample.png'), limits: { memoryMB: 16 } });
  assert.equal(res.exitCode, 0);
  assert.equal(res.output[0], 0xff); // JPEG SOI
  assert.equal(res.output[1], 0xd8);
  assert.match(res.logs, /sepia 800x500/);
});

test('AssemblyScript: an abort explains itself', async () => {
  await assert.rejects(mesh.run({ module: asModule('test-abort') }), (err) => {
    assert.equal(err.kind, 'trap');
    assert.match(err.message, /abort: Index out of range/);
    return true;
  });
});

test('distributed inference matches a JavaScript reference bit for bit', async () => {
  const { IN, HIDDEN, OUT, makeWeights, rng, batchInput } = await import('../examples/inference.js');
  const weights = makeWeights(1);
  const { id } = await mesh.putBlob(new Uint8Array(weights.buffer));
  const r = rng(2);
  const batches = [];
  for (let b = 0; b < 6; b++) {
    const x = new Float32Array(8 * IN);
    for (let i = 0; i < x.length; i++) x[i] = r();
    batches.push(x);
  }

  // Reference: the kernels' order of operations, each rounded to float32.
  const f = Math.fround;
  const w1 = weights.subarray(0, HIDDEN * IN), b1 = weights.subarray(HIDDEN * IN, HIDDEN * IN + HIDDEN);
  const w2 = weights.subarray(HIDDEN * IN + HIDDEN, HIDDEN * IN + HIDDEN + OUT * HIDDEN);
  const b2 = weights.subarray(HIDDEN * IN + HIDDEN + OUT * HIDDEN);
  const layer = (x, rows, k, w, n, b, relu) => {
    const out = new Float32Array(rows * n);
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let p = 0; p < k; p++) acc = f(acc + f(x[i * k + p] * w[j * k + p]));
        const v = f(acc + b[j]);
        out[i * n + j] = relu ? Math.max(v, 0) : v;
      }
    return out;
  };

  const nodes = new Set();
  for await (const item of mesh.map({ module: asModule('mlp'), inputs: batches.map(batchInput), blobs: [id], limits: { memoryMB: 16 } })) {
    assert.equal(item.error, undefined);
    nodes.add(item.node);
    const hidden = layer(batches[item.index], 8, IN, w1, HIDDEN, b1, true);
    const want = layer(hidden, 8, HIDDEN, w2, OUT, b2, false);
    const got = new Float32Array(item.output.buffer.slice(item.output.byteOffset, item.output.byteOffset + want.byteLength));
    assert.deepEqual(new Uint32Array(got.buffer), new Uint32Array(want.buffer), `batch ${item.index} logits differ`);
  }
  assert.ok(nodes.size > 1, 'batches ran on several nodes');
});

async function collect(gen) {
  const tokens = [];
  let stats;
  for await (const e of gen) {
    if (e.done) stats = e.stats;
    else tokens.push(e.token);
  }
  return { tokens, stats };
}

test('language model: the same text however the layers are split', async () => {
  const { model, config } = await mesh.importModel({ checkpoint: fixture('tiny-model.bin'), tokenizer: fixture('tiny-tokenizer.bin'), name: 'tiny' });
  assert.equal(config.n_layers, 4);
  // Stage servers advertise themselves every couple of seconds.
  await new Promise((r) => setTimeout(r, 2500));
  const one = await collect(mesh.generate({ model, prompt: 'hello', maxTokens: 20, stages: 1 }));
  const three = await collect(mesh.generate({ model, prompt: 'hello', maxTokens: 20, stages: 3 }));
  assert.equal(one.stats.stages.length, 1);
  assert.equal(three.stats.stages.length, 3);
  assert.equal(new Set(three.stats.stages.map((s) => s.node)).size, 3, 'three nodes');
  assert.deepEqual(three.tokens, one.tokens);
});

test('language model: a real story, when RAKKAN_LLM_DIR has one', { skip: !process.env.RAKKAN_LLM_DIR }, async () => {
  const dir = process.env.RAKKAN_LLM_DIR;
  const { model } = await mesh.importModel({ checkpoint: { path: `${dir}/stories15M.bin` }, tokenizer: { path: `${dir}/tokenizer.bin` } });
  let text = '';
  let stats;
  for await (const e of mesh.generate({ model, prompt: 'Once upon a time', maxTokens: 50, stages: 3 })) {
    if (e.done) stats = e.stats;
    else text += e.text;
  }
  assert.match(text, /^, there was a little girl named Lily/);
  assert.equal(stats.stages.length, 3);
});

test('language model: Qwen from Hugging Face, when RAKKAN_HF_MODELS has it', { skip: !process.env.RAKKAN_HF_MODELS }, async () => {
  const { model, config, chat } = await mesh.importModel({ hf: `${process.env.RAKKAN_HF_MODELS}/qwen2.5-0.5b-instruct` });
  assert.equal(config.dtype, 'bf16');
  assert.equal(chat, true);
  let text = '';
  let stats;
  for await (const e of mesh.generate({ model, prompt: 'What is the capital of France? Answer in one word.', chat: true, maxTokens: 10, stages: 3 })) {
    if (e.done) stats = e.stats;
    else text += e.text;
  }
  assert.match(text, /Paris/);
  assert.equal(stats.stages.length, 3);
});

test('no node on the socket', async () => {
  const nowhere = connect({ socket: '/nonexistent/rakkan.sock' });
  await assert.rejects(nowhere.status(), (err) => err instanceof TaskError && err.kind === 'unavailable');
});
