#!/usr/bin/env python3
"""cmpf32.py A.f32 B.f32: compare two jzdlrun/ojzdlrun dumps (int32 w,h,c + floats).
Exit 0 only if the float payloads are bit-identical. Headers are printed, not compared
(jzdlrun writes the Mat's w,h,c; ojzdlrun writes 1,1,N for the packed output)."""
import struct, sys
a, b = (open(p, 'rb').read() for p in sys.argv[1:3])
ha, hb = struct.unpack('<3i', a[:12]), struct.unpack('<3i', b[:12])
pa, pb = a[12:], b[12:]
n = len(pa) // 4
diff = sum(1 for i in range(0, min(len(pa), len(pb)), 4) if pa[i:i+4] != pb[i:i+4])
same = pa == pb
print('%s hdr %s vs %s, %d floats, %d differ, %s' % ('OK  ' if same else 'FAIL', ha, hb, n, diff + abs(len(pa) - len(pb)) // 4,
      'bit-identical' if same else 'MISMATCH'))
sys.exit(0 if same else 1)
