// Estimates π by sampling points in a square: the share falling inside the
// inscribed quarter circle tends to π/4. Tasks have no randomness, so the
// points come from a generator seeded by the input: the same input gives
// the same count on every node, which lets replicas check each other.
// Input: seed (u64), samples (u32). Output: points inside (u64).
import { input, output, readU32, readU64 } from "../assembly/index";

export function run(): i32 {
  const args = input();
  if (args.length < 12) return 1;
  let s = readU64(args, 0) | 1; // xorshift64* state, never zero
  const n = readU32(args, 8);
  let hits: u64 = 0;
  for (let i: u32 = 0; i < n; i++) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    const r = s * 0x2545F4914F6CDD1D;
    // Two 31-bit coordinates: x² + y² stays below 2^63, in integers.
    const x = r >> 33;
    const y = (r >> 2) & 0x7fffffff;
    if (x * x + y * y < (<u64>1 << 62)) hits++;
  }
  const out = new Uint8Array(8);
  store<u64>(out.dataStart, hits);
  output(out);
  return 0;
}
