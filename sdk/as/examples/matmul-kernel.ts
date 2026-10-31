// The f32 matrix product via the tensor kernel; same input and output as
// matmul-wasm.ts.
import { input, output, readU32, Tensor } from "../assembly/index";

export function run(): i32 {
  const args = input();
  const m = <i32>readU32(args, 0), k = <i32>readU32(args, 4), n = <i32>readU32(args, 8);
  const a = Tensor.fromF32(Float32Array.wrap(args.buffer, 12, m * k), m, k);
  const b = Tensor.fromF32(Float32Array.wrap(args.buffer, 12 + m * k * 4, k * n), k, n);
  output(Uint8Array.wrap(a.matmul(b).toF32().buffer));
  return 0;
}
