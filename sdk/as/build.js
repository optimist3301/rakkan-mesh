// Builds the example tasks: node build.js [example ...]
// Tasks need the stub runtime (no garbage collector: tasks are short-lived)
// and the bindings' abort instead of the "env" import the sandbox lacks.

import { execFileSync } from 'node:child_process';
import { readdirSync, mkdirSync } from 'node:fs';

// Examples, plus test modules (built into build/test-<name>.wasm).
const sources = [
  ...readdirSync('examples').filter((f) => f.endsWith('.ts')).map((f) => [`examples/${f}`, f.slice(0, -3)]),
  ...readdirSync('test').filter((f) => f.endsWith('.ts')).map((f) => [`test/${f}`, `test-${f.slice(0, -3)}`]),
].filter(([, name]) => process.argv.length <= 2 || process.argv.slice(2).includes(name));
mkdirSync('build', { recursive: true });
for (const [src, name] of sources) {
  execFileSync('npx', [
    'asc', src,
    '--outFile', `build/${name}.wasm`,
    '--runtime', 'stub',
    '--use', 'abort=assembly/index/abort',
    '-O3',
  ], { stdio: 'inherit' });
  console.log(`build/${name}.wasm`);
}
