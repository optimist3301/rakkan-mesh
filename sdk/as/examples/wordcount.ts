// Counts occurrences of a word in a byte range of attached blob 0, reading
// the blob in 64 KiB pieces: only the range is fetched to the executor.
// Run as a map over ranges, then sum the counts.
// Input: offset (u64), length (u64), then the word.
// Output: the count as a u32. A match counts in the range it starts in.
import { input, output, readU64, u32Bytes, blobs } from "../assembly/index";

const PIECE = 65536;

export function run(): i32 {
  const args = input();
  const offset = <i64>readU64(args, 0);
  const length = <i64>readU64(args, 8);
  const word = args.subarray(16);
  const w = word.length;
  if (w == 0) return 1;

  const size = blobs.size(0);
  const end = min(offset + length, size);
  // Read up to w-1 bytes past the end, for matches starting inside.
  const stop = min(end + <i64>(w - 1), size);

  // One buffer for the whole scan: the last w-1 bytes of each piece are
  // carried to the front, so every start position is checked exactly once,
  // including matches spanning two pieces.
  const buf = new Uint8Array(PIECE + w);
  let carry = 0;
  let pos = offset; // blob position of buf[carry]
  let count: u32 = 0;
  while (pos < stop) {
    const n = blobs.readInto(0, pos, buf, carry, <i32>min(<i64>PIECE, stop - pos));
    if (n <= 0) break;
    const filled = carry + n;
    const base = pos - <i64>carry; // blob position of buf[0]
    for (let i = 0; i + w <= filled; i++) {
      if (base + <i64>i >= end) break;
      let j = 0;
      while (j < w && buf[i + j] == word[j]) j++;
      if (j == w) count++;
    }
    carry = min(w - 1, filled);
    memory.copy(buf.dataStart, buf.dataStart + filled - carry, carry);
    pos += <i64>n;
  }
  output(u32Bytes(count));
  return 0;
}
