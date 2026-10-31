// Finds a 32-bit hash preimage by searching ranges in parallel across the
// mesh, stopping at the first match.
//
//   rakkan-node daemon ... -api rakkan.sock   (on this machine)
//   node examples/search.js [bits]

import { readFileSync } from 'node:fs';
import { connect } from '@rakkan/sdk';

const bits = Number(process.argv[2] ?? 28);
const mesh = connect();
const search = readFileSync(new URL('../test/fixtures/search.wasm', import.meta.url));

// The same hash as the module (wasmtest.Mix32).
function mix32(x) {
  x = Math.imul(x, 0x9e3779b1) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  return (x ^ (x >>> 13)) >>> 0;
}

const secret = Math.floor(Math.random() * 2 ** bits);
const target = mix32(secret);
const chunk = 2 ** 20;
const inputs = [];
for (let start = 0; start < 2 ** bits; start += chunk) {
  const b = Buffer.alloc(12);
  b.writeUInt32LE(start, 0);
  b.writeUInt32LE(chunk, 4);
  b.writeUInt32LE(target, 8);
  inputs.push(b);
}

const { members } = await mesh.status();
console.log(`searching 2^${bits} values in ${inputs.length} chunks over ${members.filter((m) => m.connected).length + 1} nodes`);
const started = performance.now();
const perNode = new Map();
for await (const item of mesh.map({ module: search, inputs, stopOnExitCode: 0 })) {
  if (item.error) throw item.error;
  perNode.set(item.node, (perNode.get(item.node) ?? 0) + 1);
  if (item.exitCode === 0) {
    const found = item.output.readUInt32LE(0);
    console.log(`found ${found} (expected ${secret}) in ${((performance.now() - started) / 1000).toFixed(2)}s`);
  }
}
for (const [node, n] of perNode) console.log(`  ${node.slice(-8)}: ${n} chunks`);
