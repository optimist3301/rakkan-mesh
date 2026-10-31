// @rakkan/sdk: submit WebAssembly tasks to a Rakkan mesh through the local
// node's API socket. No dependencies; Node.js 18 or newer.

import http from 'node:http';
import { createHash } from 'node:crypto';

/** A task, or the network, could not produce a result. */
export class TaskError extends Error {
  /**
   * @param {string} kind e.g. "out_of_gas", "timeout", "trap", "no_capacity"
   * @param {string} message
   * @param {number} [status] HTTP status from the node
   */
  constructor(kind, message, status) {
    super(message);
    this.name = 'TaskError';
    this.kind = kind;
    this.status = status;
  }
}

/**
 * Connects to the local node's API socket.
 * @param {{socket?: string}} [options] defaults to $RAKKAN_SOCKET or ./rakkan.sock
 */
export function connect(options = {}) {
  return new Mesh(options.socket ?? process.env.RAKKAN_SOCKET ?? 'rakkan.sock');
}

export class Mesh {
  #socket;
  #uploaded = new Set(); // hashes the node already has

  constructor(socket) {
    this.#socket = socket;
  }

  /** Uploads a module and returns its hash; repeated uploads are skipped. */
  async upload(wasm) {
    const bytes = toBytes(wasm);
    const hash = createHash('sha256').update(bytes).digest('hex');
    if (!this.#uploaded.has(hash)) {
      const res = await this.#call('POST', '/v1/modules', bytes, { type: 'application/wasm' });
      if (res.hash !== hash) throw new TaskError('internal', `node stored ${res.hash}, expected ${hash}`);
      this.#uploaded.add(hash);
    }
    return hash;
  }

  /**
   * Runs a task on the best available member and resolves to its result.
   * Rejects with a TaskError if the task fails.
   */
  async run({ module, entry, input, json, limits, replicas, signal } = {}) {
    const hash = await this.#moduleHash(module);
    const body = {
      module_hash: hash,
      entry,
      input: encodeInput(input, json),
      limits: wireLimits(limits),
      replicas,
    };
    try {
      return toResult(await this.#call('POST', '/v1/run', body, { signal }));
    } catch (err) {
      // The node may have restarted and lost the module: upload again once.
      if (err instanceof TaskError && err.kind === 'bad_request' && /unknown module/.test(err.message)) {
        this.#uploaded.delete(hash);
        body.module_hash = await this.#moduleHash(module);
        return toResult(await this.#call('POST', '/v1/run', body, { signal }));
      }
      throw err;
    }
  }

  /**
   * Runs one module over many inputs across the network, yielding results as
   * they complete: `{index, ...result}` or `{index, error}`. With
   * `stopOnExitCode`, the map ends at the first task exiting with that code.
   * Breaking out of the loop or aborting `signal` cancels the remaining tasks.
   */
  async *map({ module, inputs, entry, limits, concurrency, stopOnExitCode, signal } = {}) {
    const hash = await this.#moduleHash(module);
    const body = {
      module_hash: hash,
      entry,
      inputs: inputs.map((i) => encodeInput(i)),
      limits: wireLimits(limits),
      concurrency,
      stop_on_exit_code: stopOnExitCode,
    };
    const controller = new AbortController();
    const onAbort = () => controller.abort(signal.reason);
    signal?.addEventListener('abort', onAbort, { once: true });
    const res = await this.#request('POST', '/v1/map', JSON.stringify(body), { type: 'application/json', signal: controller.signal });
    try {
      if (res.statusCode !== 200) throw await errorFrom(res);
      let buffered = '';
      for await (const chunk of res) {
        buffered += chunk;
        let nl;
        while ((nl = buffered.indexOf('\n')) >= 0) {
          const line = buffered.slice(0, nl);
          buffered = buffered.slice(nl + 1);
          if (!line) continue;
          const msg = JSON.parse(line);
          if (msg.done) return;
          if (msg.error) {
            yield { index: msg.index, error: new TaskError(msg.error.kind, msg.error.message) };
          } else {
            yield { index: msg.index, ...toResult(msg) };
          }
        }
      }
    } finally {
      // Closing the connection early cancels tasks still running.
      signal?.removeEventListener('abort', onAbort);
      if (!res.complete) controller.abort();
    }
  }

  /** The node's identity, load and view of the network. */
  async status() {
    return this.#call('GET', '/v1/status');
  }

  async #moduleHash(module) {
    if (typeof module === 'string' && /^[0-9a-f]{64}$/.test(module)) return module;
    if (module == null) throw new TypeError('module is required: bytes or a hash from upload()');
    return this.upload(module);
  }

  async #call(method, path, body, { type, signal } = {}) {
    let payload = body;
    if (body !== undefined && !(body instanceof Uint8Array)) {
      payload = JSON.stringify(body);
      type = 'application/json';
    }
    const res = await this.#request(method, path, payload, { type, signal });
    if (res.statusCode < 200 || res.statusCode >= 300) throw await errorFrom(res);
    return JSON.parse(await readAll(res));
  }

  #request(method, path, payload, { type, signal } = {}) {
    return new Promise((resolve, reject) => {
      const req = http.request(
        { socketPath: this.#socket, method, path, signal, headers: type ? { 'content-type': type } : {} },
        resolve,
      );
      req.on('error', (err) => {
        if (err.code === 'ENOENT' || err.code === 'ECONNREFUSED') {
          reject(new TaskError('unavailable', `no Rakkan node listening on ${this.#socket}: ${err.message}`));
        } else {
          reject(err);
        }
      });
      req.end(payload);
    });
  }
}

function toBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (typeof data === 'string') return Buffer.from(data);
  throw new TypeError('expected bytes, an ArrayBuffer or a string');
}

function encodeInput(input, json) {
  if (json !== undefined) return Buffer.from(JSON.stringify(json)).toString('base64');
  if (input == null) return undefined;
  return Buffer.from(toBytes(input)).toString('base64');
}

function wireLimits(l) {
  if (!l) return undefined;
  return {
    gas: l.gas,
    timeout_ms: l.timeoutMs,
    memory_mb: l.memoryMB,
    max_output_bytes: l.maxOutputBytes,
    max_input_bytes: l.maxInputBytes,
  };
}

function toResult(r) {
  return {
    exitCode: r.exit_code,
    output: Buffer.from(r.output ?? '', 'base64'),
    logs: r.logs ?? '',
    gasUsed: r.gas_used,
    elapsedMs: r.elapsed_ms,
    node: r.node,
    attempts: r.attempts,
    replicas: (r.replicas ?? []).map((x) => ({ node: x.node, agreed: x.agreed, error: x.error })),
  };
}

async function readAll(res) {
  let s = '';
  for await (const chunk of res) s += chunk;
  return s;
}

async function errorFrom(res) {
  const text = await readAll(res);
  try {
    const { error } = JSON.parse(text);
    return new TaskError(error.kind, error.message, res.statusCode);
  } catch {
    return new TaskError('internal', `HTTP ${res.statusCode}: ${text}`, res.statusCode);
  }
}
