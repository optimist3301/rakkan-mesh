// The same f32 matrix product as the tensor kernel, in plain
// AssemblyScript, for comparison: input m, k, n (u32), then a[m,k] and
// b[k,n]; output c[m,n].
import { input, output, readU32 } from "../assembly/index";

export function run(): i32 {
  const args = input();
  const m = <i32>readU32(args, 0), k = <i32>readU32(args, 4), n = <i32>readU32(args, 8);
  const a = Float32Array.wrap(args.buffer, 12, m * k);
  const b = Float32Array.wrap(args.buffer, 12 + m * k * 4, k * n);
  const c = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc: f32 = 0;
      for (let p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
      c[i * n + j] = acc;
    }
  }
  output(Uint8Array.wrap(c.buffer));
  return 0;
}
