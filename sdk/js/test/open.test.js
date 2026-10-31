// An open network: two certified members and a guest that joined by proof
// of work, with the policy learned from the bootstrap node.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { connect } from '../index.js';
import { startCluster } from './cluster.js';

const echo = readFileSync(new URL('fixtures/echo.wasm', import.meta.url));

let cluster;
let member;
let guest;

before(async () => {
  // The guest has the most free slots, so members' tasks go to it first.
  cluster = await startCluster({ nodes: 2, workers: 1, guests: 1, guestWorkers: 4 });
  member = connect({ socket: cluster.sockets[0] });
  guest = connect({ socket: cluster.sockets[2] });
});

after(() => cluster?.stop());

test('a guest joins an open network by proof of work', async () => {
  const st = await guest.status();
  assert.equal(st.guest, true);
  assert.equal(st.open, true);
  assert.equal(st.pow_bits, 12);
  const seen = (await member.status()).members.find((m) => m.id === cluster.ids[2]);
  assert.equal(seen.guest, true);
  assert.deepEqual(seen.roles, ['guest']);
  assert.equal(typeof seen.score, 'number');
});

test('a guest\'s results are checked by a member', async () => {
  const res = await member.run({ module: echo, input: 'trust, but verify' });
  assert.equal(res.output.toString(), 'trust, but verify');
  assert.equal(res.node, cluster.ids[2]);
  assert.equal(res.checked, true);
  assert.equal(res.replicas.length, 2);
  assert.ok(res.replicas.every((r) => r.agreed));
  const ledger = await member.trust();
  const g = ledger.find((p) => p.id === cluster.ids[2]);
  assert.equal(g.member, false);
  assert.ok(g.verified >= 1);
  assert.ok(g.consumed > 0, 'the guest\'s work is credited to it');
  assert.equal(g.cheated, 0);
});

test('a guest runs tasks on credit', async () => {
  const res = await guest.run({ module: echo, input: 'on credit' });
  assert.equal(res.output.toString(), 'on credit');
  assert.equal(res.checked, false, 'the guest trusts members and itself');
});
