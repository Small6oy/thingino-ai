"""Walk a venus .bin of the ReLU/QSUM kind (thingino-ai 416x320 detector): every W block in file
order, with its form (dense 32x32 groups or per-channel), shape fields and the param record."""
import struct
import numpy as np


def i32(b, o):
    return struct.unpack("<i", b[o:o + 4])[0]


SIG = b"\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00"


def layers(blob):
    """W header [bytes, 7, nob, nic, kh, kw, 4, 32, icg] + W data, followed by the param record"""
    out = []
    o = 36
    n = len(blob)
    while o + 36 <= n:
        h = struct.unpack("<9i", blob[o - 36:o])
        if h[1] == 7 and h[3] >= 1 and h[6] == 4 and h[7] == 32 and h[8] in (1, 32) and 0 < h[0] <= n - o \
                and h[0] % (32 if h[8] == 1 else 256) == 0 and blob[o + h[0]:o + h[0] + 10] == SIG:
            nbytes, _, nob, nic, kh, kw, _, _, icg = h
            out.append(dict(w=o, nbytes=nbytes, nob=nob, nic=nic, kh=kh, kw=kw, dw=icg == 1, p=o + nbytes))
            o += nbytes
            continue
        o += 1
    return out


def bt(blob, L, k):
    p = L["p"] + 0x25 + 256 * k
    return np.frombuffer(blob[p:p + 128], "<i4"), np.frombuffer(blob[p + 128:p + 256], "<i4")
