#!/usr/bin/env python3
"""v4probe.py: compare every int8 blob of a JZDL model against libjzdl (QEMU).

  v4probe.py MODEL.bin FRAME.rgb WORKDIR [--run] [--blobs a,b,...]

--run builds one identity-head probe model per int8 blob (probe.py) and runs
them all through jzdlrun under QEMU (env JZDL_QEMU, XB as in ../ojzdl/gate.sh),
6 at a time. Without --run it reuses WORKDIR's outputs. Then it runs
jzdl_ref.forward and prints, per layer, how many values of its top blob match.
This is how version-4 (yolov5) behaviour is pinned down: the first layer that
mismatches is the one whose hypothesis in jzdl_ref.py is wrong.
"""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jzdl, probe, jzdl_ref


def blob_bits(m):
    """activation width of every int8 blob, walking producers"""
    bits = {m.input_blob: 8}
    for L in m.layers:
        if L.type == 16:
            continue
        if L.type == 46:
            b = L.params[16]
        elif L.type in (53, 73, 74):
            b = L.params[15]
        elif L.type == 78:
            b = L.params[0]
        elif L.type in (51, 76):
            b = None
        else:
            b = bits[L.bottoms[0]]
        for t in L.tops:
            bits[t] = b
    return bits


def main(a):
    model, frame, work = a[1], a[2], a[3]
    m = jzdl.Model.parse(open(model, 'rb').read())
    bits = blob_bits(m)
    want = sorted(b for b, v in bits.items() if v is not None and b != m.input_blob)
    if '--blobs' in a:
        want = [int(x) for x in a[a.index('--blobs') + 1].split(',')]
    os.makedirs(work, exist_ok=True)
    if '--run' in a:
        q = os.environ.get('JZDL_QEMU', os.path.expanduser('~/jzdl-qemu'))
        xb = os.environ.get('XB', os.path.expanduser('~/xb/tf-v3/output/xb'))
        sysroot = xb + '/host/mipsel-buildroot-linux-uclibc/sysroot'
        cmds = []
        for b in want:
            p = probe.make_probe(m, b, bits[b])
            pb = '%s/p%d.bin' % (work, b)
            open(pb, 'wb').write(p.to_bytes())
            cmds.append('%s/fbuild/qemu-mipsel -cpu XBurstR2 -L %s -E LD_LIBRARY_PATH=%s/run %s/run/jzdlrun %s %s %s/p%d.out > %s/p%d.log 2>&1'
                        % (q, sysroot, q, q, pb, frame, work, b, work, b))
        open(work + '/cmds.sh', 'w').write('\n'.join(cmds) + '\n')
        subprocess.run('xargs -P 6 -I{} sh -c "{}" < %s/cmds.sh' % work, shell=True, check=False)
    rgb = np.fromfile(frame, np.uint8).reshape(m.in_h, m.in_w, m.in_c)

    def lib(b):
        if b == m.input_blob:
            return rgb.astype(np.int64) - 128
        f = '%s/p%d.out.-999.f32' % (work, b)
        if not os.path.exists(f):
            return None
        w, h, c = np.fromfile(f, np.int32, count=3)
        return np.fromfile(f, np.float32, offset=12).reshape(h, w, c).astype(np.int64)

    if '--isolated' in a:
        # every layer fed libjzdl's own inputs: tests one layer's rule at a time
        jzdl_ref.VERSION[0] = m.version
        for i, L in enumerate(m.layers):
            if L.type in (16, 51, 76):
                continue
            ins = [lib(b) for b in L.bottoms]
            got = lib(L.tops[0])
            if any(x is None for x in ins) or got is None:
                continue
            try:
                r = jzdl_ref.forward_layer(m, L, ins, bits)[0]
            except Exception as e:
                print('%3d t%-3d: reference error %r' % (i, L.type, e))
                continue
            if r.shape != got.shape:
                print('%3d t%-3d blob %3d: SHAPE ref %s lib %s' % (i, L.type, L.tops[0], r.shape, got.shape))
                continue
            d = r - got
            eq = (d == 0).mean()
            print('%3d t%-3d blob %3d: %s %7.3f %% equal, max |diff| %d  params %s' % (
                i, L.type, L.tops[0], 'OK  ' if eq == 1 else 'DIFF', 100 * eq, abs(d).max(), ' '.join(map(str, L.params))))
        return
    try:
        ref = jzdl_ref.forward(m, rgb)
    except Exception as e:  # keep going: report how far the reference got
        print('reference stopped:', repr(e))
        ref = getattr(jzdl_ref, 'LAST_BLOBS', {})
    for i, L in enumerate(m.layers):
        for b in L.tops:
            if b not in want:
                continue
            f = '%s/p%d.out.-999.f32' % (work, b)
            if not os.path.exists(f):
                print('%3d t%-3d blob %3d: no libjzdl output' % (i, L.type, b))
                continue
            w, h, c = np.fromfile(f, np.int32, count=3)
            got = np.fromfile(f, np.float32, offset=12).reshape(h, w, c).astype(np.int64)
            r = ref.get(b)
            if r is None:
                print('%3d t%-3d blob %3d: lib %s range %d..%d, no reference' % (i, L.type, b, got.shape, got.min(), got.max()))
                continue
            if r.shape != got.shape:
                print('%3d t%-3d blob %3d: SHAPE ref %s lib %s' % (i, L.type, b, r.shape, got.shape))
                continue
            eq = (r == got).mean()
            print('%3d t%-3d blob %3d: %s %7.3f %% equal  (bits %s, lib %d..%d)' % (
                i, L.type, b, 'OK  ' if eq == 1 else 'DIFF', 100 * eq, bits[b], got.min(), got.max()))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    main(sys.argv)
