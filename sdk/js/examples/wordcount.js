// Counts a word in a large text file across the mesh: the file is uploaded
// once as a blob and attached to every task; each task scans one range of
// it, so executors fetch only the chunks of their ranges. A map/reduce:
// the counts are summed here.
//
//   node examples/wordcount.js <file> <word> [parts]

import { readFileSync, statSync } from 'node:fs';
import { connect } from '@rakkan/sdk';

const [file, word, partsArg] = process.argv.slice(2);
if (!file || !word) {
  console.error('usage: node examples/wordcount.js <file> <word> [parts]');
  process.exit(2);
}
const mesh = connect();
const module = readFileSync(new URL('../../as/build/wordcount.wasm', import.meta.url));

const size = statSync(file).size;
const parts = Number(partsArg ?? 32);
const step = Math.ceil(size / parts);
const { id } = await mesh.putBlob({ path: file });

const inputs = [];
for (let off = 0; off < size; off += step) {
  const b = Buffer.alloc(16 + Buffer.byteLength(word));
  b.writeBigUInt64LE(BigInt(off), 0);
  b.writeBigUInt64LE(BigInt(step), 8);
  b.write(word, 16);
  inputs.push(b);
}

const started = performance.now();
let total = 0;
const nodes = new Map();
for await (const item of mesh.map({ module, inputs, blobs: [id] })) {
  if (item.error) throw item.error;
  total += item.output.readUInt32LE(0);
  nodes.set(item.node, (nodes.get(item.node) ?? 0) + 1);
}
console.log(`"${word}": ${total} in ${(size / 1e6).toFixed(1)} MB, ${inputs.length} parts, ${((performance.now() - started) / 1000).toFixed(2)}s`);
for (const [node, n] of nodes) console.log(`  ${node.slice(-8)}: ${n} parts`);
