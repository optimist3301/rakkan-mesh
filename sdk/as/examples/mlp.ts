// Classifies a batch with a two-layer perceptron whose weights live in
// attached blob 0: logits = relu(x W1^T + b1) W2^T + b2. The matrix work
// runs in the host's tensor kernels; weights go from the blob to the host
// without passing through the task's memory.
//
// Blob 0 layout (f32, little-endian): W1 [hidden, in], b1 [hidden],
// W2 [out, hidden], b2 [out].
// Input: in, hidden, out, rows (u32 each), then rows x in f32 inputs.
// Output: rows x out f32 logits, then rows i32 predicted classes.
import { input, output, readU32, Tensor, DType } from "../assembly/index";

export function run(): i32 {
  const args = input();
  const nIn = <i32>readU32(args, 0);
  const nHidden = <i32>readU32(args, 4);
  const nOut = <i32>readU32(args, 8);
  const rows = <i32>readU32(args, 12);
  const x = Float32Array.wrap(args.buffer, 16, rows * nIn);

  let off: i64 = 0;
  const w1 = Tensor.fromBlob(0, off, DType.F32, nHidden, nIn);
  off += <i64>(nHidden * nIn * 4);
  const b1 = Tensor.fromBlob(0, off, DType.F32, 1, nHidden);
  off += <i64>(nHidden * 4);
  const w2 = Tensor.fromBlob(0, off, DType.F32, nOut, nHidden);
  off += <i64>(nOut * nHidden * 4);
  const b2 = Tensor.fromBlob(0, off, DType.F32, 1, nOut);

  const hidden = Tensor.fromF32(x, rows, nIn).matmulT(w1).add(b1).relu();
  const logits = hidden.matmulT(w2).add(b2);

  output(Uint8Array.wrap(logits.toF32().buffer));
  output(Uint8Array.wrap(logits.argmax().buffer));
  return 0;
}
