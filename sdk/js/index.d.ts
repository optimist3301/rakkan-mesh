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
  /** The output; empty with `outputAsBlob` when the output came as a blob. */
  output: Buffer;
  /** Set when the output exceeded 1 MiB and travelled as a blob. */
  outputBlob?: string;
  outputSize: number;
  logs: string;
  gasUsed: number;
  elapsedMs: number;
  /** Peer ID of the member that ran the task. */
  node: string;
  /** Executors tried; more than 1 means failover happened. */
  attempts: number;
  replicas: Replica[];
  /**
   * In an open network: the task ran on a guest, and an independent run
   * (on a member where possible) confirmed the outcome.
   */
  checked: boolean;
}

export interface RunOptions {
  /** Module bytes, or a hash returned by upload(). */
  module: Bytes;
  entry?: string;
  input?: Bytes;
  /** Encoded as the input instead of `input`. */
  json?: unknown;
  /** A blob from putBlob() as the input. */
  inputBlob?: string;
  /** Up to 16 stored blobs the task reads in pieces with blob_read. */
  blobs?: string[];
  /** Leave a large output in the node's store instead of downloading it. */
  outputAsBlob?: boolean;
  limits?: Limits;
  /** Run on this many distinct members and accept the majority outcome. */
  replicas?: number;
  signal?: AbortSignal;
}

export interface MapOptions {
  module: Bytes;
  inputs?: Bytes[];
  /** Blobs from putBlob(), instead of `inputs`. */
  inputBlobs?: string[];
  /** Blobs attached to every task: shared data, fetched once per executor. */
  blobs?: string[];
  outputAsBlob?: boolean;
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
  /** Admitted by proof of work rather than certificate. */
  guest?: boolean;
  /** A guest's reputation, 0 to 1. */
  score?: number;
  connected: boolean;
  relayed: boolean;
  rtt_ms: number;
  load?: Load;
}

export interface Load {
  slots: number;
  busy: number;
  queued: number;
  /** Optional host APIs, e.g. "image/1". */
  capabilities: string[];
}

export interface Status {
  id: string;
  network: string;
  /** This node joined by proof of work, without a certificate. */
  guest: boolean;
  /** The network admits guests, proving `pow_bits` of work. */
  open: boolean;
  pow_bits?: number;
  load: Load;
  members: Member[];
}

export class TaskError extends Error {
  /** e.g. "out_of_gas", "timeout", "trap", "no_capacity", "no_consensus", "unavailable" */
  kind: string;
  status?: number;
}

export type BlobSource = Bytes | NodeJS.ReadableStream | ReadableStream<Uint8Array> | { path: string };

export class Mesh {
  upload(wasm: Bytes): Promise<string>;
  putBlob(data: BlobSource, options?: { signal?: AbortSignal }): Promise<{ id: string; size: number }>;
  getBlob(id: string, options?: { start?: number; end?: number; signal?: AbortSignal }): Promise<NodeJS.ReadableStream>;
  readBlob(id: string, options?: { start?: number; end?: number; signal?: AbortSignal }): Promise<Buffer>;
  run(options: RunOptions): Promise<Result>;
  map(options: MapOptions): AsyncGenerator<MapItem, void, undefined>;
  status(): Promise<Status>;
  trust(): Promise<TrustPeer[]>;
  /**
   * Imports a llama2.c checkpoint and tokenizer, or with `hf` a Hugging Face
   * model directory (config.json, tokenizer.json, *.safetensors): Llama,
   * Mistral, Qwen2/2.5, Qwen3.
   */
  importModel(options: { checkpoint: BlobSource; tokenizer: BlobSource; name?: string; signal?: AbortSignal } | { hf: string; name?: string; signal?: AbortSignal }): Promise<{ model: string; config: ModelConfig; chat?: boolean }>;
  generate(options: GenerateOptions): AsyncGenerator<GenerateEvent, void, undefined>;
}

export interface TrustPeer {
  id: string;
  /** This node or a certified member: trusted without checks. */
  member: boolean;
  /** Reputation from 0 to 1 (EigenTrust); 1 for members. */
  score: number;
  /** Gas this node spent for the peer. */
  served: number;
  /** Gas the peer spent for this node. */
  consumed: number;
  /** Results confirmed by independent runs. */
  verified: number;
  failed: number;
  /** Wrong results caught. */
  cheated: number;
  lastCheated?: Date;
}

export interface ModelConfig {
  dim: number;
  hidden_dim: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  vocab_size: number;
  seq_len: number;
  head_dim?: number;
  rope_theta?: number;
  qkv_bias?: boolean;
  qk_norm?: boolean;
  /** How weight matrices are stored: "f32" or "bf16". */
  dtype?: string;
}

export interface GenerateOptions {
  model: string;
  prompt: string;
  /** Wrap the prompt in the model's chat format (instruction-tuned models). */
  chat?: boolean;
  maxTokens?: number;
  /** 0 (default) always picks the most likely token. */
  temperature?: number;
  /** Reproduces a sampled generation exactly. */
  seed?: number;
  /** Pipeline stages to split the layers into; default one per serving member. */
  stages?: number;
  signal?: AbortSignal;
}

export interface GenerateStats {
  promptTokens: number;
  tokens: number;
  elapsedMs: number;
  tokensPerSecond: number;
  /** `witness` is the trusted node checking a stage that ran on a guest. */
  stages: { from: number; to: number; node: string; witness?: string }[];
  /** Stages moved to another node after a failure, mid-generation. */
  failovers: number;
  /** Guests caught returning wrong activations; their witness took over. */
  caught: number;
}

export type GenerateEvent = { token: number; text: string; done?: undefined } | { done: true; stats: GenerateStats };

export function connect(options?: { socket?: string }): Mesh;
