// Batch inference across the mesh: a two-layer perceptron's weights are
// uploaded once as a blob and attached to every task; batches of inputs
// spread over the members as a map. The matrix work runs in the tensor
// kernels (capability tensor/1), bit-identical on every node.
//
//   node examples/inference.js [samples] [batch]

import { readFileSync } from 'node:fs';
import { connect } from '@rakkan/sdk';

export const IN = 784, HIDDEN = 256, OUT = 10;

/** A deterministic PRNG, so weights and inputs are reproducible. */
export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0;
    return (s / 2 ** 32 - 0.5) * 0.2;
  };
}

/** Weights in the layout mlp.wasm expects: W1, b1, W2, b2 as f32. */
export function makeWeights(seed) {
  const r = rng(seed);
  const w = new Float32Array(HIDDEN * IN + HIDDEN + OUT * HIDDEN + OUT);
  for (let i = 0; i < w.length; i++) w[i] = r();
  return w;
}

/** Task input: dimensions, then rows x IN f32 samples. */
export function batchInput(samples) {
  const rows = samples.length / IN;
  const b = Buffer.alloc(16 + samples.length * 4);
  [IN, HIDDEN, OUT, rows].forEach((v, i) => b.writeUInt32LE(v, 4 * i));
  Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength).copy(b, 16);
  return b;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const samples = Number(process.argv[2] ?? 4096);
  const batch = Number(process.argv[3] ?? 256);
  const mesh = connect();
  const module = readFileSync(new URL('../../as/build/mlp.wasm', import.meta.url));
  const weights = makeWeights(42);
  const { id } = await mesh.putBlob(new Uint8Array(weights.buffer));

  const r = rng(7);
  const inputs = [];
  for (let s = 0; s < samples; s += batch) {
    const x = new Float32Array(Math.min(batch, samples - s) * IN);
    for (let i = 0; i < x.length; i++) x[i] = r();
    inputs.push(batchInput(x));
  }

  const started = performance.now();
  const counts = new Array(OUT).fill(0);
  const nodes = new Map();
  for await (const item of mesh.map({ module, inputs, blobs: [id], limits: { memoryMB: 64 } })) {
    if (item.error) throw item.error;
    const rows = (inputs[item.index].length - 16) / 4 / IN;
    for (let i = 0; i < rows; i++) counts[item.output.readInt32LE(rows * OUT * 4 + 4 * i)]++;
    nodes.set(item.node, (nodes.get(item.node) ?? 0) + 1);
  }
  const s = (performance.now() - started) / 1000;
  console.log(`${samples} samples in ${inputs.length} batches: ${s.toFixed(2)}s, ${(samples / s).toFixed(0)} samples/s`);
  console.log(`predicted classes: ${counts.join(' ')}`);
  for (const [node, n] of nodes) console.log(`  ${node.slice(-8)}: ${n} batches`);
}
