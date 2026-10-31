// Starts a local Rakkan network of real daemons for integration tests.

import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { connect } from '../index.js';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

/** Builds rakkan-node unless RAKKAN_BIN points at one. */
export function binary(dir) {
  if (process.env.RAKKAN_BIN) return process.env.RAKKAN_BIN;
  const bin = join(dir, 'rakkan-node');
  execFileSync('go', ['build', '-o', bin, './cmd/rakkan-node'], { cwd: repoRoot, stdio: 'inherit' });
  return bin;
}

/**
 * Starts `nodes` certified members, then `guests` nodes joining by proof of
 * work; guests make the root open the network first. `guestWorkers`
 * overrides the guests' worker count.
 */
export async function startCluster({ nodes = 3, workers = 2, llm = false, guests = 0, guestWorkers = workers } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'rakkan-sdk-'));
  const bin = binary(dir);
  const run = (...args) => execFileSync(bin, args, { cwd: dir, stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();

  const network = run('network', 'init', '-out', 'root.key');
  if (guests > 0) run('network', 'open', '-pow', '12', '-policy', 'policy.json');
  const procs = [];
  const sockets = [];
  const ids = [];
  let bootstrap = '';
  for (let i = 0; i < nodes + guests; i++) {
    const guest = i >= nodes;
    const id = run('id', '-key', `n${i}.key`);
    ids.push(id);
    // A guest has no certificate: it asks the bootstrap node for the
    // policy and proves the work it demands.
    if (!guest) run('network', 'issue', '-peer', id, '-out', `n${i}.cert`);
    const socket = join(dir, `n${i}.sock`);
    const args = [
      'daemon', '-network', network, '-key', `n${i}.key`, '-cert', `n${i}.cert`,
      '-proof', `n${i}.proof`, '-data', `data${i}`,
      '-listen', '/ip4/127.0.0.1/udp/0/quic-v1', '-mdns=false', '-nat=false', '-peers=',
      '-workers', String(guest ? guestWorkers : workers), '-api', socket,
    ];
    if (guests > 0 && i === 0) args.push('-policy', 'policy.json');
    if (bootstrap) args.push('-bootstrap', bootstrap);
    if (llm) args.push('-llm');
    const p = spawn(bin, args, { cwd: dir, stdio: ['ignore', 'pipe', 'pipe'] });
    procs.push(p);
    p.stderr.resume();
    const addr = await new Promise((resolveAddr, reject) => {
      let out = '';
      p.stdout.on('data', (d) => {
        out += d;
        const m = out.match(/listening on (\S+)/);
        if (m && out.includes('api on')) resolveAddr(m[1]);
      });
      p.on('exit', (code) => reject(new Error(`daemon ${i} exited with ${code}`)));
    });
    if (!bootstrap) bootstrap = addr;
    sockets.push(socket);
  }

  const stop = () => procs.forEach((p) => p.kill());
  // Wait until the first node sees every other member's load.
  const mesh = connect({ socket: sockets[0] });
  const deadline = Date.now() + 20_000;
  for (;;) {
    const st = existsSync(sockets[0]) ? await mesh.status().catch(() => null) : null;
    if (st && st.members.filter((m) => m.connected && m.load).length === nodes + guests - 1) break;
    if (Date.now() > deadline) {
      stop();
      throw new Error(`cluster did not form: ${JSON.stringify(st)}`);
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  // kill(i) stops daemon i abruptly, as a crash would.
  const kill = (i) => procs[i].kill('SIGKILL');
  return { dir, sockets, ids, stop, kill };
}
