// Estimates π across the mesh by Monte Carlo: a map of independent tasks,
// each sampling points from its own seed, reduced here by summing.
//
//   node examples/pi.js [tasks] [samples per task]

import { readFileSync } from 'node:fs';
import { connect } from '@rakkan/sdk';

const tasks = Number(process.argv[2] ?? 64);
const perTask = Number(process.argv[3] ?? 10_000_000);
const mesh = connect();
const module = readFileSync(new URL('../../as/build/montecarlo.wasm', import.meta.url));

// Distinct seeds: the task's index, spread over 64 bits.
const inputs = [];
for (let i = 0; i < tasks; i++) {
  const b = Buffer.alloc(12);
  b.writeBigUInt64LE((BigInt(i + 1) * 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn, 0);
  b.writeUInt32LE(perTask, 8);
  inputs.push(b);
}

const started = performance.now();
let hits = 0n;
const nodes = new Map();
for await (const item of mesh.map({ module, inputs })) {
  if (item.error) throw item.error;
  hits += item.output.readBigUInt64LE(0);
  nodes.set(item.node, (nodes.get(item.node) ?? 0) + 1);
}
const pi = (4 * Number(hits)) / (tasks * perTask);
console.log(`π ≈ ${pi.toFixed(6)} (error ${Math.abs(pi - Math.PI).toExponential(1)}) from ${tasks * perTask} points in ${((performance.now() - started) / 1000).toFixed(2)}s`);
for (const [node, n] of nodes) console.log(`  ${node.slice(-8)}: ${n} tasks`);
