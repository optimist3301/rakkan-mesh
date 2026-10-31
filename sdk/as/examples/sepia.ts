// Tones a photo sepia and outputs a JPEG at most 800 pixels wide: decoding,
// scaling and encoding use the host's image kernels; the per-pixel colour
// transform is ordinary AssemblyScript over the pixel buffer.
import { Image, log } from "../assembly/index";

export function run(): i32 {
  const photo = Image.decodeInput();
  if (photo === null) return 1;
  const img = photo.fit(800, 800);
  if (img === null) return 2;
  photo.free();

  const px = img.readPixels();
  for (let i = 0; i < px.length; i += 4) {
    const r = <u32>px[i], g = <u32>px[i + 1], b = <u32>px[i + 2];
    // Classic sepia matrix in 1/1024 fixed point: integer, deterministic.
    px[i] = <u8>min((r * 402 + g * 788 + b * 194) >> 10, 255);
    px[i + 1] = <u8>min((r * 357 + g * 702 + b * 172) >> 10, 255);
    px[i + 2] = <u8>min((r * 279 + g * 547 + b * 134) >> 10, 255);
  }
  img.writePixels(px);
  log("sepia " + img.width.toString() + "x" + img.height.toString());
  return img.outputJPEG(85) > 0 ? 0 : 3;
}
