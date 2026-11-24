#!/usr/bin/env python3
"""mxu2dis.py DECODE OBJDUMP.txt: annotate objdump -d output with MXU2 mnemonics.

DECODE is QEMU's target/mips/tcg/mxu2.decode (gtxaspec fork); it is read at run
time, not copied. Lines whose word matches an MXU2 pattern get '; MXU2 NAME fields'.
Field values are raw (vd/vs/vt = vector regs, base/rt = GPRs, df 0..3 = b,h,w,d).
"""
import re, sys

def load(path):
    fmts, pats = {}, []
    for line in open(path):
        line = line.split('#')[0].rstrip()
        if not line or line[0] in '&%':
            continue
        toks = line.split()
        name, rest = toks[0], toks[1:]
        bits, fmt, extra = '', None, []
        for t in rest:
            if t.startswith('@'):
                fmt = t
            elif t.startswith('&') or '=' in t and not re.match(r'^[a-z_]+:', t):
                extra.append(t)
            else:
                bits += t + ' '
        if name.startswith('@'):
            fmts[name] = bits
            continue
        if fmt:
            # overlay: explicit bits from the pattern, fields from the format
            pb, fb = bits.split(), fmts[fmt].split()
            pats.append((name, merge(pb, fb)))
        else:
            pats.append((name, parse(bits.split())))
    return pats

def parse(toks):
    """-> list of 32 entries: '0'/'1'/'.' or (field, bitindex-from-msb-of-field, width)"""
    out = []
    for t in toks:
        m = re.match(r'^([a-z_0-9]+):(s?)(\d+)$', t)
        if m:
            w = int(m.group(3))
            out += [(m.group(1), i, w) for i in range(w)]
        else:
            out += list(t)
    assert len(out) == 32, (toks, len(out))
    return out

def merge(pb, fb):
    p, f = parse(pb), parse(fb)
    return [p[i] if p[i] in '01' else f[i] for i in range(32)]

def match(pats, w):
    for name, bits in pats:
        ok, fields = True, {}
        for i, b in enumerate(bits):
            v = (w >> (31 - i)) & 1
            if b in ('0', '1'):
                if int(b) != v:
                    ok = False
                    break
            elif isinstance(b, tuple):
                fields[b[0]] = fields.get(b[0], 0) << 1 | v
        if ok:
            return name, fields
    return None

if __name__ == '__main__':
    pats = load(sys.argv[1])
    for line in open(sys.argv[2]):
        line = line.rstrip('\n')
        m = re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s', line)
        if m:
            r = match(pats, int(m.group(2), 16))
            if r:
                line = '%-60s ; MXU2 %s %s' % (line[:60], r[0], ' '.join('%s=%d' % kv for kv in r[1].items()))
        print(line)
