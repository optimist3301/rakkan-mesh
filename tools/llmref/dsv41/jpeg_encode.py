"""A small baseline JPEG encoder for test images Pillow's encoder does not
make: any sampling factors per component (4:4:0, 4:1:1, ...), Adobe
markers with any transform (RGB, YCCK), no JFIF. The decoder under test is
Pillow's (libjpeg-turbo); what this encodes only has to be valid.

Huffman tables hold only the symbols used, each code of the same length,
which is a valid canonical table.
"""
import struct

import numpy as np

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]

_C = np.array([[np.sqrt((1 if k == 0 else 2) / 8) * np.cos((2 * n + 1) * k * np.pi / 16) for n in range(8)] for k in range(8)])


def _fdct(b):
    return _C @ b @ _C.T


def _category(v):
    v = abs(int(v))
    return v.bit_length()


def _bits(v, n):
    v = int(v)
    return v if v >= 0 else v + (1 << n) - 1


class _BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, code, length):
        for i in range(length - 1, -1, -1):
            self.acc = (self.acc << 1) | ((code >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc = self.n = 0

    def flush(self):
        while self.n:
            self.put(1, 1)
        return bytes(self.out)


def _table(symbols):
    """Fixed-length codes for the symbols (sorted): the length leaves the
    all-ones code unused."""
    symbols = sorted(set(symbols)) or [0]
    length = max(1, (len(symbols)).bit_length())
    return {s: (i, length) for i, s in enumerate(symbols)}, length


def encode(planes, factors, ids, quant=8, adobe=None, jfif=False):
    """planes: one uint8 array [H, W] per component, all of the image's
    size; factors: (h, v) per component; ids: component identifiers;
    adobe: None or the APP14 transform byte. Components are downsampled by
    averaging to their factors and coded in one interleaved scan."""
    H, W = planes[0].shape
    hmax = max(f[0] for f in factors)
    vmax = max(f[1] for f in factors)
    mcux, mcuy = -(-W // (8 * hmax)), -(-H // (8 * vmax))
    q = np.full(64, quant, dtype=np.int64)
    comps = []
    for p, (h, v) in zip(planes, factors):
        sx, sy = hmax // h, vmax // v
        full = np.pad(p.astype(np.float64), ((0, mcuy * 8 * vmax - H), (0, mcux * 8 * hmax - W)), mode="edge")
        small = full.reshape(full.shape[0] // sy, sy, full.shape[1] // sx, sx).mean(axis=(1, 3))
        comps.append(small - 128)
    # Coefficients, block by block in scan order.
    blocks = []
    for my in range(mcuy):
        for mx in range(mcux):
            for ci, (h, v) in enumerate(factors):
                for by in range(v):
                    for bx in range(h):
                        y0, x0 = (my * v + by) * 8, (mx * h + bx) * 8
                        c = _fdct(comps[ci][y0 : y0 + 8, x0 : x0 + 8]).flatten()
                        z = np.round(c[ZIGZAG] / q).astype(np.int64)
                        blocks.append((ci, z))
    # Symbols, then tables, then the scan.
    dcs, acs, pred = [], [], [0] * len(planes)
    for ci, z in blocks:
        diff = z[0] - pred[ci]
        pred[ci] = z[0]
        dcs.append(_category(diff))
        run = 0
        for k in range(1, 64):
            if z[k] == 0:
                run += 1
                continue
            while run > 15:
                acs.append(0xF0)
                run -= 16
            acs.append(run << 4 | _category(z[k]))
            run = 0
        if run:
            acs.append(0x00)
    dct, _ = _table(dcs)
    act, _ = _table(acs)
    w = _BitWriter()
    pred = [0] * len(planes)
    for ci, z in blocks:
        diff = z[0] - pred[ci]
        pred[ci] = z[0]
        n = _category(diff)
        w.put(*dct[n])
        if n:
            w.put(_bits(diff, n), n)
        run = 0
        for k in range(1, 64):
            if z[k] == 0:
                run += 1
                continue
            while run > 15:
                w.put(*act[0xF0])
                run -= 16
            n = _category(z[k])
            w.put(*act[run << 4 | n])
            w.put(_bits(z[k], n), n)
            run = 0
        if run:
            w.put(*act[0x00])
    data = w.flush()

    def seg(marker, body):
        return struct.pack(">BBH", 0xFF, marker, len(body) + 2) + body

    def dht(cls, table):
        counts = [0] * 16
        syms = sorted(table, key=lambda s: table[s][0])
        for s in syms:
            counts[table[s][1] - 1] += 1
        return seg(0xC4, bytes([cls << 4]) + bytes(counts) + bytes(syms))

    out = b"\xff\xd8"
    if jfif:
        out += seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if adobe is not None:
        out += seg(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    out += seg(0xDB, bytes([0]) + bytes(int(x) for x in q))
    sof = struct.pack(">BHHB", 8, H, W, len(planes))
    for cid, (h, v) in zip(ids, factors):
        sof += bytes([cid, h << 4 | v, 0])
    out += seg(0xC0, sof)
    out += dht(0, dct) + dht(1, act)
    sos = bytes([len(planes)])
    for cid in ids:
        sos += bytes([cid, 0x00])
    sos += bytes([0, 63, 0])
    out += seg(0xDA, sos) + data + b"\xff\xd9"
    return out
