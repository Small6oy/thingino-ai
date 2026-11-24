#!/usr/bin/env python3
"""Where each F pixel of a layer's NNDWR 0x20/0x21 stream sits in the layer's input map.

Replays the T41 rules of QEMU hw/misc/ingenic-nna.c (nn_rwr, nn_fwrite, nn_cmd tile moves,
nn_mac addressing) with the F RAM holding *stream pixel indices* instead of bytes. Every
in-bounds tap of every MAC then names (stream pixel, plane) -> input (y, x).

usage: fmap.py <dir> <segment>   -> <dir>/seg<N>_fmap.npy: int32 [n_pixels][3] = y, x, plane
                                    (-1 where the MACs never read that pixel)
                                    <dir>/seg<N>_tiles.npy: per CPOC (fx, fy, rows2, sx, sy,
                                    pixel mask, streamed W entry of the last MAC)
"""
import sys

import numpy as np

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])
LINES = 1024
SNAKE = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
MASK = {0x20: 0x1, 0x21: 0x3, 0x23: 0xf, 0x28: 0x3, 0x29: 0xfff, 0x2a: 0xff, 0x2b: 0xfff,
        0x2c: 0x7ff, 0x2d: 0xffff, 0x2e: 0x7ff, 0x2f: 0xfff, 0x30: 0xffff, 0x31: 0xffff,
        0x32: 0x1ff, 0x33: 0x1, 0x34: 0x1ff, 0x37: 0x1fc, 0x38: 0x13, 0x39: 0x3c, 0x3a: 0x11,
        0x3b: 0xff, 0x3c: 0xf, 0x3d: 0xf, 0x3e: 0x1ff, 0x3f: 0xff, 0x42: 0xff, 0x43: 0x3,
        0x44: 0x7, 0x45: 0xf, 0x46: 0x3, 0x47: 0x1f, 0x48: 0xffff, 0x49: 0xffff, 0x50: 0x3ff,
        0x51: 0x3ff, 0x52: 0x3ff, 0x53: 0x3ff, 0x54: 0x3, 0x55: 0x3, 0x58: 0xf, 0x59: 0x7fff,
        0x5a: 0x77, 0x5b: 0x3, 0x5c: 0xffff, 0x5e: 0xff}
RESET = {0x20: 1, 0x23: 1, 0x29: 0xfff, 0x2b: 0xfff, 0x2c: 0x7ff, 0x2e: 0x7ff, 0x33: 1,
         0x37: 0x50, 0x38: 0x12, 0x39: 0x14, 0x3e: 0x1ff, 0x3f: 0xff, 0x48: 0x3210, 0x49: 0x7654}


def sx12(v):
    v &= 0xfff
    return v - 0x1000 if v & 0x800 else v


class Engine:
    def __init__(self):
        self.reg = [0] * 0x80
        for k, v in RESET.items():
            self.reg[k] = v
        self.wfcw = [0] * 16
        self.wfcw_ptr = 0
        self.line = [0, 0]
        self.half = [0, 0]
        self.slot = [0, 0]
        self.dirty = True
        self.nslots = [0, 0]
        self.inc = [[], []]
        self.jump = [[], []]
        self.fram = np.full(LINES * 4, -1, np.int64)   # 32-byte pixel slots (vec 32 only)
        self.npx = 0
        self.wr_w = self.wr_r = self.wr_base = 0
        self.wsrc = {}            # W RAM entry -> index of the streamed 256-B entry
        self.nent = 0
        self.last_mac = None
        self.wuse = set()         # (streamed W entry, input plane, ky, kx) the MACs used
        self.acc = [[] for _ in range(8)]
        self.ocb = None

    def vec(self):
        return 8 << self.reg[0x55]

    def unroll(self, p, lo, hi, depth):
        prog = self.wfcw[8 * p:8 * p + 8]
        k = lo
        while k < hi and depth < 4:
            ctrl = (prog[k] >> 11) & 0xf
            inc = prog[k] & 0x3ff
            inc = inc - 0x400 if inc & 0x200 else inc
            if ctrl == 0xf:
                lpend = (prog[k] >> 8) & 7
                for _ in range((prog[k] & 0xff) + 1):
                    self.unroll(p, k + 1, lpend, depth + 1)
                k = max(lpend, k + 1)
                continue
            for _ in range(max(ctrl, 1)):
                if len(self.inc[p]) == 1024:
                    return
                self.inc[p].append(inc)
                self.jump[p].append(not ctrl)
            k += 1

    def slots(self):
        if not self.dirty:
            return
        for p in range(2):
            n = (self.reg[0x5a] >> (4 * p)) & 7
            self.inc[p], self.jump[p] = [], []
            if n:
                self.unroll(p, 0, n, 0)
            else:
                self.inc[p], self.jump[p] = [1], [False]
        self.dirty = False

    def rwr(self, r, v):
        v &= MASK.get(r, 0xffffffff)
        self.reg[r] = v
        if r in (0x50, 0x51):
            p = r & 1
            self.line[p] = v if r == 0x50 else (self.reg[0x50] + v) % LINES
            self.half[p] = self.slot[p] = 0
        elif r == 0x58:
            self.wfcw_ptr = v
        elif r == 0x59:
            self.wfcw[self.wfcw_ptr & 15] = v
            self.wfcw_ptr += 1
            self.dirty = True
        elif r == 0x5a:
            self.dirty = True
        elif r == 0x30:
            self.wr_base = self.wr_r = v
            self.reg[0x34] = 0
        elif r == 0x34:
            self.wr_base += v
            self.wr_r = self.wr_base
        elif r == 0x5c:
            self.wr_w = v

    def wwrite(self, imm):
        if imm & 1:
            self.wsrc[self.wr_w % 1024] = self.nent
            self.nent += 1
            self.wr_w += 1

    def fwrite(self, p):
        """one 64-byte NNDWR 0x20/0x21: 64 / vec pixels of the stream"""
        nb = self.vec()
        per = 64 // nb
        base = self.line[p] * 128 + self.half[p] * 64
        for i in range(per):
            self.fram[((base + i * nb) % (LINES * 128)) // nb] = self.npx
            self.npx += 1
        self.half[p] += 1
        if self.half[p] * 64 < 4 * nb:
            return
        self.half[p] = 0
        self.slots()
        n = len(self.inc[p])
        while True:
            k = self.slot[p] % n
            self.slot[p] += 1
            self.line[p] = (self.line[p] + self.inc[p][k]) % LINES
            if not (self.jump[p][self.slot[p] % n] and self.slot[p] % n):
                break

    def cmd(self, imm, tiles=None):
        if imm == 0:
            keep = getattr(self, "wuse", set())
            kp = getattr(self, "pairs", [])
            self.__init__()
            self.wuse = keep
            self.pairs = kp
            return
        if imm & 1 and tiles is not None:
            tiles.append(self.last_mac)
            self.pairs.append([list(a) for a in self.acc])
        if imm & 2:
            self.acc = [[] for _ in range(8)]
        mv = (imm >> 5) & 7
        rows2 = bool(self.reg[0x38] & 0x10)
        if mv == 5:
            d = (2 if rows2 else 1) * (2 if self.reg[0x3a] & 0x10 else 1)
            self.reg[0x29] = (self.reg[0x29] + d) & 0xfff
            self.reg[0x2a] = (self.reg[0x2a] + d) & 0xff
        elif mv in (6, 7):
            self.reg[0x29] = (self.reg[0x29] - self.reg[0x2a]) & 0xfff
            self.reg[0x2a] = 0
            if mv == 6:
                self.reg[0x2b] = (self.reg[0x2b] + (4 if rows2 else 8) *
                                  (2 if self.reg[0x3a] & 1 else 1)) & 0xfff

    def mac(self, word, out, imm=0):
        """QEMU nn_mac (T41) addressing. Records, per output pixel, every (W entry, F source)
        product the engine adds, so a CPOC's sums can be recomputed from the weights."""
        R = self.reg
        ntaps = bin(R[0x3e] & 0x1ff).count("1")
        sl = (((R[0x3b] >> 3) & 7) + 2) // 2
        self.last_mac = (sx12(R[0x2b]), sx12(R[0x29]), int(bool(R[0x38] & 0x10)),
                         2 if R[0x3a] & 1 else 1, 2 if R[0x3a] & 0x10 else 1, R[0x3f],
                         self.wsrc.get(self.wr_r % 1024, -1))
        sx = 2 if R[0x3a] & 1 else 1
        sy = 2 if R[0x3a] & 0x10 else 1
        planes = R[0x3c] + 1
        rows2 = bool(R[0x38] & 0x10)
        fx, fy, fw, fh = sx12(R[0x2b]), sx12(R[0x29]), R[0x2e], R[0x2c]
        nb = self.vec()
        pl_px = R[0x37] * nb // 8
        taps = [g for g in range(9) if R[0x3e] & (1 << (SNAKE[g][0] * 3 + SNAKE[g][1]))]
        for i, g in enumerate(taps):
            ky, kx = SNAKE[g]
            for k in range(planes):
                ent = self.wsrc.get((self.wr_r + (k * ntaps + i) * sl) % 1024, -1)
                self.wuse.add((ent, k, ky, kx))
                for p in range(8):
                    if not R[0x3f] & (1 << p):
                        continue
                    r, c = (p >> 2, p & 3) if rows2 else (0, p)
                    x, y = fx + c * sx + kx, fy + r * sy + ky
                    if x < 0 or x > fw or y < 0 or y > fh:
                        self.acc[p].append((ent, -1))          # pad (0x2f)
                        continue
                    px = word + (r * sy + ky) * R[0x39] + c * sx + kx + k * pl_px
                    o = ((px >> 2) * 128 + (px & 3) * nb) % (LINES * 128)
                    src = self.fram[o // nb]
                    self.acc[p].append((ent, src))
                    if src >= 0:
                        out.append((src, y, x, k))
        if imm & 1:
            self.wr_r += ntaps * planes * sl

def main():
    d, seg = sys.argv[1], int(sys.argv[2])
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    r = list(np.load(f"{d}/resets.npy")) + [len(ops)]
    s = ops[r[seg]:r[seg + 1]]
    w0 = s["vec"][:, :4].copy().view("<u4")[:, 0]
    e = Engine()
    e.pairs = []
    out = []
    tiles = []
    for i in range(len(s)):
        op, imm = int(s["op"][i]), int(s["imm"][i])
        if op == 0:
            e.rwr(imm, int(w0[i]))
        elif op == 2 and imm in (0x20, 0x21):
            e.fwrite(imm & 1)
        elif op == 2 and imm < 0x20:
            e.wwrite(imm)
        elif op == 4:
            e.mac(int(w0[i]), out, imm)
        elif op == 5:
            e.cmd(imm, tiles)
    a = np.array(out, np.int64)
    m = np.full((e.npx, 3), -1, np.int32)
    conflict = 0
    for src, y, x, k in a:
        if m[src, 0] >= 0 and tuple(m[src]) != (y, x, k):
            conflict += 1
        m[src] = (y, x, k)
    used = (m[:, 0] >= 0).sum()
    print(f"segment {seg}: {e.npx} stream pixels, {len(a)} MAC reads, {used} pixels read,"
          f" {conflict} conflicting reads")
    np.save(f"{d}/seg{seg}_fmap.npy", m)
    # per CPOC: fx, fy, rows2, sx, sy, pixel mask, streamed W entry the MAC started at
    np.save(f"{d}/seg{seg}_tiles.npy", np.array(tiles, np.int64))
    np.save(f"{d}/seg{seg}_wuse.npy", np.array(sorted(e.wuse), np.int64))
    # per CPOC k, pixel p: the (W entry, F source or -1 = pad) products -> flat (k, p, ent, src)
    flat = [(k, p, ent, src) for k, px in enumerate(e.pairs) for p, lst in enumerate(px)
            for ent, src in lst]
    np.save(f"{d}/seg{seg}_pairs.npy", np.array(flat, np.int64).reshape(-1, 4))
    print(f"  {len(tiles)} CPOC tiles recorded")


if __name__ == "__main__":
    main()
