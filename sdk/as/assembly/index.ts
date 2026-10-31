// AssemblyScript bindings for the Rakkan task ABI.
//
// A task exports `run(): i32` (0 means success) and uses these functions
// for its input, output, logs, attached blobs and image kernels. Build with
// build.js, which passes the flags tasks need: the stub runtime, and this
// module's abort in place of the default import from "env", which the
// sandbox does not provide.

// --- host functions ---------------------------------------------------------

@external("rakkan", "input_len") declare function input_len(): u32;
@external("rakkan", "input_read") declare function input_read(dst: usize, offset: u32, len: u32): u32;
@external("rakkan", "output_write") declare function output_write(src: usize, len: u32): void;
@external("rakkan", "log") declare function host_log(src: usize, len: u32): void;
@external("rakkan", "blob_count") declare function blob_count(): i32;
@external("rakkan", "blob_size") declare function blob_size(index: i32): i64;
@external("rakkan", "blob_read") declare function blob_read(index: i32, dst: usize, offset: i64, len: i32): i32;

// --- input and output -------------------------------------------------------

/** The task's input. */
export function input(): Uint8Array {
  const n = input_len();
  const buf = new Uint8Array(n);
  input_read(buf.dataStart, 0, n);
  return buf;
}

/** The task's input as UTF-8 text. */
export function inputString(): string {
  return String.UTF8.decode(input().buffer);
}

/** Appends bytes to the task's output. */
export function output(data: Uint8Array): void {
  output_write(data.dataStart, data.length);
}

/** Appends UTF-8 text to the task's output. */
export function outputString(s: string): void {
  const b = String.UTF8.encode(s);
  output_write(changetype<usize>(b), b.byteLength);
}

/** Writes a line to the task's diagnostic log. */
export function log(message: string): void {
  const b = String.UTF8.encode(message + "\n");
  host_log(changetype<usize>(b), b.byteLength);
}

/** Reads a little-endian u32 at offset. */
export function readU32(buf: Uint8Array, offset: i32): u32 {
  return load<u32>(buf.dataStart + offset);
}

/** Reads a little-endian u64 at offset. */
export function readU64(buf: Uint8Array, offset: i32): u64 {
  return load<u64>(buf.dataStart + offset);
}

/** Bytes of a little-endian u32. */
export function u32Bytes(v: u32): Uint8Array {
  const b = new Uint8Array(4);
  store<u32>(b.dataStart, v);
  return b;
}

// --- attached blobs ---------------------------------------------------------

/**
 * Blobs attached to the task, read in pieces: the node fetches only the
 * chunks read, so a task may use a small part of a very large blob.
 */
export namespace blobs {
  /** Number of attached blobs. */
  export function count(): i32 {
    return blob_count();
  }

  /** Size of blob index in bytes, or -1 if there is no such blob. */
  export function size(index: i32): i64 {
    return blob_size(index);
  }

  /** Reads into buf from offset; returns the bytes read, 0 at the end. */
  export function read(index: i32, offset: i64, buf: Uint8Array): i32 {
    return blob_read(index, buf.dataStart, offset, buf.length);
  }

  /**
   * Reads up to len bytes from offset into buf at position at, without
   * allocating: use it in loops (the stub runtime has no garbage collector).
   */
  export function readInto(index: i32, offset: i64, buf: Uint8Array, at: i32, len: i32): i32 {
    if (at < 0 || len < 0 || at + len > buf.length) return -1;
    return blob_read(index, buf.dataStart + at, offset, len);
  }
}

// --- image kernels (capability image/1) -------------------------------------

@external("rakkan_image_v1", "decode_input") declare function img_decode_input(): i32;
@external("rakkan_image_v1", "decode_blob") declare function img_decode_blob(index: i32): i32;
@external("rakkan_image_v1", "decode") declare function img_decode(src: usize, len: u32): i32;
@external("rakkan_image_v1", "create") declare function img_create(w: i32, h: i32): i32;
@external("rakkan_image_v1", "width") declare function img_width(h: i32): i32;
@external("rakkan_image_v1", "height") declare function img_height(h: i32): i32;
@external("rakkan_image_v1", "resize") declare function img_resize(h: i32, w: i32, ht: i32): i32;
@external("rakkan_image_v1", "fit") declare function img_fit(h: i32, w: i32, ht: i32): i32;
@external("rakkan_image_v1", "crop") declare function img_crop(h: i32, x: i32, y: i32, w: i32, ht: i32): i32;
@external("rakkan_image_v1", "grayscale") declare function img_grayscale(h: i32): i32;
@external("rakkan_image_v1", "blur") declare function img_blur(h: i32, r: i32): i32;
@external("rakkan_image_v1", "pixels_read") declare function img_pixels_read(h: i32, dst: usize, len: u32): i32;
@external("rakkan_image_v1", "pixels_write") declare function img_pixels_write(h: i32, src: usize, len: u32): i32;
@external("rakkan_image_v1", "encode_output") declare function img_encode_output(h: i32, format: i32, quality: i32): i32;
@external("rakkan_image_v1", "free") declare function img_free(h: i32): i32;

function image(handle: i32): Image | null {
  return handle < 0 ? null : new Image(handle);
}

/**
 * An image held by the host. Operations run as native code in the sandboxed
 * worker and return a new image, or null on failure (bad arguments,
 * undecodable data, or over the task's memory budget).
 */
export class Image {
  constructor(public readonly handle: i32) {}

  /** Decodes the task's input as PNG or JPEG. */
  static decodeInput(): Image | null {
    return image(img_decode_input());
  }

  /** Decodes attached blob index. */
  static decodeBlob(index: i32): Image | null {
    return image(img_decode_blob(index));
  }

  static decode(data: Uint8Array): Image | null {
    return image(img_decode(data.dataStart, data.length));
  }

  /** A new transparent image. */
  static create(width: i32, height: i32): Image | null {
    return image(img_create(width, height));
  }

  get width(): i32 {
    return img_width(this.handle);
  }

  get height(): i32 {
    return img_height(this.handle);
  }

  resize(width: i32, height: i32): Image | null {
    return image(img_resize(this.handle, width, height));
  }

  /** Scales down to fit maxWidth x maxHeight, keeping the aspect ratio. */
  fit(maxWidth: i32, maxHeight: i32): Image | null {
    return image(img_fit(this.handle, maxWidth, maxHeight));
  }

  crop(x: i32, y: i32, width: i32, height: i32): Image | null {
    return image(img_crop(this.handle, x, y, width, height));
  }

  grayscale(): Image | null {
    return image(img_grayscale(this.handle));
  }

  blur(radius: i32): Image | null {
    return image(img_blur(this.handle, radius));
  }

  /** The pixels: RGBA, 8 bits per channel, non-premultiplied, row by row. */
  readPixels(): Uint8Array {
    const px = new Uint8Array(this.width * this.height * 4);
    img_pixels_read(this.handle, px.dataStart, px.length);
    return px;
  }

  writePixels(px: Uint8Array): bool {
    return img_pixels_write(this.handle, px.dataStart, px.length) >= 0;
  }

  /** Appends the image to the task output as PNG; returns the bytes written. */
  outputPNG(): i32 {
    return img_encode_output(this.handle, 0, 0);
  }

  /** Appends the image to the task output as JPEG (quality 1-100). */
  outputJPEG(quality: i32): i32 {
    return img_encode_output(this.handle, 1, quality);
  }

  /** Releases the image and its share of the memory budget. */
  free(): void {
    img_free(this.handle);
  }
}

// --- runtime ----------------------------------------------------------------

/**
 * Replaces AssemblyScript's default abort (imported from "env", which the
 * sandbox does not offer): logs the failure and traps, failing the task.
 */
export function abort(message: string | null, fileName: string | null, line: u32, column: u32): void {
  log(
    "abort: " + (message !== null ? message : "") + " at " +
      (fileName !== null ? fileName : "?") + ":" + line.toString() + ":" + column.toString(),
  );
  unreachable();
}
