/// <reference types="node" />

export type Bytes = Uint8Array | ArrayBuffer | string;

export interface Limits {
  /** Deterministic instruction budget. */
  gas?: number;
  timeoutMs?: number;
  memoryMB?: number;
  maxOutputBytes?: number;
  maxInputBytes?: number;
}

export interface Replica {
  node: string;
  /** Whether this replica's outcome matched the accepted one. */
  agreed: boolean;
  error?: string;
}

export interface Result {
  exitCode: number;
  output: Buffer;
  logs: string;
  gasUsed: number;
  elapsedMs: number;
  /** Peer ID of the member that ran the task. */
  node: string;
  /** Executors tried; more than 1 means failover happened. */
  attempts: number;
  replicas: Replica[];
}

export interface RunOptions {
  /** Module bytes, or a hash returned by upload(). */
  module: Bytes;
  entry?: string;
  input?: Bytes;
  /** Encoded as the input instead of `input`. */
  json?: unknown;
  limits?: Limits;
  /** Run on this many distinct members and accept the majority outcome. */
  replicas?: number;
  signal?: AbortSignal;
}

export interface MapOptions {
  module: Bytes;
  inputs: Bytes[];
  entry?: string;
  limits?: Limits;
  concurrency?: number;
  /** End the map at the first task exiting with this code. */
  stopOnExitCode?: number;
  signal?: AbortSignal;
}

export type MapItem = ({ index: number; error?: undefined } & Result) | { index: number; error: TaskError };

export interface Member {
  id: string;
  roles: string[];
  connected: boolean;
  relayed: boolean;
  rtt_ms: number;
  load?: { slots: number; busy: number; queued: number };
}

export interface Status {
  id: string;
  network: string;
  load: { slots: number; busy: number; queued: number };
  members: Member[];
}

export class TaskError extends Error {
  /** e.g. "out_of_gas", "timeout", "trap", "no_capacity", "no_consensus", "unavailable" */
  kind: string;
  status?: number;
}

export class Mesh {
  upload(wasm: Bytes): Promise<string>;
  run(options: RunOptions): Promise<Result>;
  map(options: MapOptions): AsyncGenerator<MapItem, void, undefined>;
  status(): Promise<Status>;
}

export function connect(options?: { socket?: string }): Mesh;
