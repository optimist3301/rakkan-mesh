// Finds a 32-bit hash preimage in a range: the same task as the
// hand-assembled search.wasm, written in AssemblyScript.
// Input: start, count, target as little-endian u32s.
// Output: the preimage as a u32, exit 0; exit 1 if none in the range.
import { input, output, readU32, u32Bytes } from "../assembly/index";

function mix32(x: u32): u32 {
  x *= 0x9e3779b1;
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  return x;
}

export function run(): i32 {
  const args = input();
  const start = readU32(args, 0);
  const end = start + readU32(args, 4);
  const target = readU32(args, 8);
  for (let x = start; x < end; x++) {
    if (mix32(x) == target) {
      output(u32Bytes(x));
      return 0;
    }
  }
  return 1;
}
