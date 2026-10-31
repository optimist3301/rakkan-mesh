// Makes JPEG thumbnails of PNG and JPEG files across the mesh: decoding,
// resizing and encoding run as native image kernels inside the sandboxed
// workers of whichever members offer the "image/1" capability.
//
//   node examples/thumbnails.js photo1.jpg photo2.png ...

import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
import { connect } from '@rakkan/sdk';

const mesh = connect();
const module = readFileSync(new URL('../test/fixtures/thumbnail-jpeg.wasm', import.meta.url));
const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('usage: node examples/thumbnails.js <image>...');
  process.exit(2);
}

// Upload each photo as a blob: streamed, chunked, verified, fetched by the
// executor that gets the task.
const blobs = [];
for (const f of files) blobs.push((await mesh.putBlob({ path: f })).id);

const started = performance.now();
for await (const item of mesh.map({ module, inputBlobs: blobs, limits: { memoryMB: 64 } })) {
  const name = basename(files[item.index]);
  if (item.error) {
    console.error(`${name}: ${item.error.kind}: ${item.error.message}`);
  } else if (item.exitCode !== 0) {
    console.error(`${name}: not a decodable image (exit ${item.exitCode})`);
  } else {
    writeFileSync(`thumb-${name.replace(/\.[^.]+$/, "")}.jpg`, item.output);
    console.log(`${name}: ${item.output.length} bytes, on ${item.node.slice(-8)}, gas ${item.gasUsed}`);
  }
}
console.log(`done in ${((performance.now() - started) / 1000).toFixed(2)}s`);
