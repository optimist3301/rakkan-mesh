// @rakkan/sdk: submit WebAssembly tasks to a Rakkan mesh through the local
// node's API socket. No dependencies; Node.js 18 or newer.

import http from 'node:http';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { Readable } from 'node:stream';

// Inputs larger than this are uploaded as blobs rather than sent as base64.
const BLOB_THRESHOLD = 256 * 1024;

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
   * Stores data in the node's blob store, streaming it, and resolves to
   * `{id, size}`. Accepts bytes, a string, a Node or web readable stream, or
   * `{path}` to stream a file.
   */
  async putBlob(data, { signal } = {}) {
    let body = data;
    if (data && typeof data === 'object' && typeof data.path === 'string') body = createReadStream(data.path);
    else if (typeof ReadableStream !== 'undefined' && data instanceof ReadableStream) body = Readable.fromWeb(data);
    else if (!(data instanceof Readable)) body = toBytes(data);
    const res = await this.#request('POST', '/v1/blobs', body, { type: 'application/octet-stream', signal });
    if (res.statusCode !== 200) throw await errorFrom(res);
    return JSON.parse(await readAll(res));
  }

  /**
   * Streams a blob from the node. `start`/`end` (inclusive) select a byte
   * range, e.g. to resume a download.
   */
  async getBlob(id, { start, end, signal } = {}) {
    const range = start != null || end != null ? `bytes=${start ?? 0}-${end ?? ''}` : undefined;
    const res = await this.#request('GET', `/v1/blobs/${id}`, undefined, { signal, headers: range ? { range } : {} });
    if (res.statusCode !== 200 && res.statusCode !== 206) throw await errorFrom(res);
    return res;
  }

  /** Reads a whole blob into a Buffer. */
  async readBlob(id, options) {
    const chunks = [];
    for await (const c of await this.getBlob(id, options)) chunks.push(c);
    return Buffer.concat(chunks);
  }

  /**
   * Runs a task on the best available member and resolves to its result.
   * Rejects with a TaskError if the task fails. Large inputs travel as
   * blobs; give `inputBlob` for data already stored with putBlob(). A large
   * output arrives as a blob and is downloaded into `output` unless
   * `outputAsBlob` is set, in which case only `outputBlob` is given.
   */
  async run({ module, entry, input, json, inputBlob, limits, replicas, outputAsBlob, signal } = {}) {
    const hash = await this.#moduleHash(module);
    const body = {
      module_hash: hash,
      entry,
      limits: wireLimits(limits),
      replicas,
      ...(await this.#inputField(input, json, inputBlob, signal)),
    };
    let res;
    try {
      res = await this.#call('POST', '/v1/run', body, { signal });
    } catch (err) {
      // The node may have restarted and lost the module: upload again once.
      if (err instanceof TaskError && err.kind === 'bad_request' && /unknown module/.test(err.message)) {
        this.#uploaded.delete(hash);
        body.module_hash = await this.#moduleHash(module);
        res = await this.#call('POST', '/v1/run', body, { signal });
      } else {
        throw err;
      }
    }
    return this.#result(res, outputAsBlob, signal);
  }

  async #inputField(input, json, inputBlob, signal) {
    if (inputBlob) return { input_blob: inputBlob };
    if (json !== undefined) input = JSON.stringify(json);
    if (input == null) return {};
    const bytes = toBytes(input);
    if (bytes.length > BLOB_THRESHOLD) return { input_blob: (await this.putBlob(bytes, { signal })).id };
    return { input: Buffer.from(bytes).toString('base64') };
  }

  async #result(r, outputAsBlob, signal) {
    const res = toResult(r);
    if (res.outputBlob && !outputAsBlob) res.output = await this.readBlob(res.outputBlob, { signal });
    return res;
  }

  /**
   * Runs one module over many inputs across the network, yielding results as
   * they complete: `{index, ...result}` or `{index, error}`. With
   * `stopOnExitCode`, the map ends at the first task exiting with that code.
   * Breaking out of the loop or aborting `signal` cancels the remaining tasks.
   */
  async *map({ module, inputs, inputBlobs, entry, limits, concurrency, stopOnExitCode, outputAsBlob, signal } = {}) {
    const hash = await this.#moduleHash(module);
    const body = {
      module_hash: hash,
      entry,
      limits: wireLimits(limits),
      concurrency,
      stop_on_exit_code: stopOnExitCode,
    };
    if (inputBlobs) {
      body.input_blobs = inputBlobs;
    } else if (inputs.some((i) => toBytes(i).length > BLOB_THRESHOLD)) {
      // Large inputs go as blobs; the rest do as well, for one request shape.
      body.input_blobs = [];
      for (const i of inputs) body.input_blobs.push((await this.putBlob(i, { signal })).id);
    } else {
      body.inputs = inputs.map((i) => Buffer.from(toBytes(i)).toString('base64'));
    }
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
            yield { index: msg.index, ...(await this.#result(msg, outputAsBlob, signal)) };
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

  #request(method, path, payload, { type, signal, headers = {} } = {}) {
    return new Promise((resolve, reject) => {
      const req = http.request(
        { socketPath: this.#socket, method, path, signal, headers: type ? { ...headers, 'content-type': type } : headers },
        resolve,
      );
      req.on('error', (err) => {
        if (err.code === 'ENOENT' || err.code === 'ECONNREFUSED') {
          reject(new TaskError('unavailable', `no Rakkan node listening on ${this.#socket}: ${err.message}`));
        } else {
          reject(err);
        }
      });
      if (payload instanceof Readable) {
        payload.on('error', (err) => req.destroy(err));
        payload.pipe(req);
      } else {
        req.end(payload);
      }
    });
  }
}

function toBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (typeof data === 'string') return Buffer.from(data);
  throw new TypeError('expected bytes, an ArrayBuffer or a string');
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
    outputBlob: r.output_blob,
    outputSize: r.output_size,
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
