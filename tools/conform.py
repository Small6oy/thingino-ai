#!/usr/bin/env python3
# Copyright (C) 2026 Thingino Project
# SPDX-License-Identifier: MIT
"""One command: is this model the same everywhere it runs?

  conform.py MODEL_DIR [--frames DIR] [--update-card] [--json OUT]

MODEL_DIR is a model library folder (model.bin + model.json, e.g. t31-jzdl/<family>/v<N>/ or
t41-venus/...) or a build_model.py output folder (model_t31.bin, model_t41.bin, checkpoint.pt).
What runs depends on what is there and on this machine; every check prints PASS, FAIL or SKIP (why).

T31 (JZDL .bin):
  export   checkpoint.pt re-exported by train/ == model.bin, byte for byte
  ojzdl    tools/jzdl/jzdl_ref.py == tools/ojzdl/ojzdlrun (native) on every frame
  libjzdl  tools/ojzdl/gate.sh: libjzdl under QEMU == ojzdl (mips, native) == jzdl_ref
           (needs the offline runtime: JZDL_QEMU, XB; see gate.sh)
T41 (venus .bin):
  onnaq    tools/onna/onnaq -e cpu, graph from the .bin (no ONNX, no .qgraph): runs, heads
  oracle   onnaq -e oracle -c (every conv on the NNA driver vs the CPU conv; ORACLE=1 build)
  executor tools/nnaop/m416/mgkcmp.sh: onnaq == Ingenic's PC executor, head by head
           (needs model.onnx + model_quant.mgk, Docker with the magik image, mgkdump)

Frames: five deterministic 416x320 RGB frames made here (black, white, checker, gradient, noise
seed 0), plus every *.rgb in --frames (real images are not shipped in this repo). --update-card
writes the golden output md5s per frame into model.json ("conformance"), so a camera can check
itself after a flash. Exit status 1 when any check FAILs.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
H, W = 320, 416
results = []


def report(check, status, detail=""):
    results.append(dict(check=check, status=status, detail=detail))
    print("%-9s %-4s %s" % (check, status, detail), flush=True)


def frames(dirs, out):
    """the synthetic frames (+ real ones): {name: path} of 416x320x3 RGB bytes"""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    synth = {
        "black": np.zeros((H, W, 3), np.uint8),
        "white": np.full((H, W, 3), 255, np.uint8),
        "checker": (((yy // 16 + xx // 16) % 2) * 255).astype(np.uint8)[..., None].repeat(3, 2),
        "gradient": np.stack([xx * 255 // (W - 1), yy * 255 // (H - 1), (xx + yy) * 255 // (W + H - 2)], 2).astype(np.uint8),
        "noise": rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
    }
    fr = {}
    for k, a in synth.items():
        p = os.path.join(out, k + ".rgb")
        a.tofile(p)
        fr[k] = p
    for d in dirs or []:
        for f in sorted(os.listdir(d)):
            if f.endswith(".rgb") and os.path.getsize(os.path.join(d, f)) == H * W * 3:
                fr[f[:-4]] = os.path.join(d, f)
    return fr


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def payload(p):
    return open(p, "rb").read()[12:]          # jzdlrun / ojzdlrun / jzdl_ref .f32: int32 w, h, c, then floats


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def build(target, d):
    r = run(["make", "-s", target], cwd=d)
    return r.returncode == 0 and os.path.exists(os.path.join(d, target))


def card_of(mdir):
    p = os.path.join(mdir, "model.json")
    return (json.load(open(p)), p) if os.path.exists(p) else (None, None)


# ------------------------------------------------------------------ T31
def t31(mdir, binp, fr, card):
    golden = {}
    ck = os.path.join(mdir, "checkpoint.pt")
    if not os.path.exists(ck):
        report("export", "SKIP", "no checkpoint.pt")
    else:
        try:
            import torch
            sys.path[:0] = [os.path.join(REPO, "train"), os.path.join(REPO, "train", "qat"), os.path.join(REPO, "tools", "jzdl")]
            import train as T
            T.setup_exact_math()
            c = torch.load(ck, map_location="cpu", weights_only=True)
            classes = (card or {}).get("classes") or c.get("classes") or T.CLASSES
            arch = c.get("arch", "persondet")
            T.CLASSES, T.NC = list(classes), len(classes)
            net = T.build_net(arch, "persondet" if arch == "yolon" else "scratch")
            net.load_state_dict(c["net"])
            net.eval()
            tmp = os.path.join(os.path.dirname(fr["black"]), "reexport.bin")
            T.export_any(net, tmp)
            same = open(tmp, "rb").read() == open(binp, "rb").read()
            report("export", "PASS" if same else "FAIL", "%s (%s, %d classes) re-export %s model.bin" % (
                os.path.basename(ck), arch, len(classes), "==" if same else "!="))
        except SystemExit as e:                       # e.g. the YOLO-N template is not on this machine
            report("export", "SKIP", str(e))
        except Exception as e:                        # noqa: BLE001 (report, do not crash the other checks)
            report("export", "FAIL", "%s: %s" % (type(e).__name__, e))
    oj = os.path.join(REPO, "tools", "ojzdl")
    if not build("ojzdlrun", oj):
        report("ojzdl", "SKIP", "cannot build tools/ojzdl/ojzdlrun")
        return golden
    ref = os.path.join(REPO, "tools", "jzdl", "jzdl_ref.py")
    bad = []
    for name, f in fr.items():
        t = os.path.join(os.path.dirname(fr["black"]), "t31_" + name)
        r1 = run([sys.executable, ref, binp, f, t + ".ref.f32"])
        r2 = run([os.path.join(oj, "ojzdlrun"), binp, f, t + ".oj"])
        if r1.returncode or r2.returncode or not os.path.exists(t + ".oj.-999.f32"):
            bad.append("%s (run failed: %s)" % (name, (r1.stdout + r2.stdout).strip().splitlines()[-1:]))
            continue
        a, b = payload(t + ".ref.f32"), payload(t + ".oj.-999.f32")
        if a != b:
            bad.append(name)
        golden[name] = hashlib.md5(a).hexdigest()
    report("ojzdl", "FAIL" if bad else "PASS", ("differs on " + ", ".join(bad)) if bad else
           "jzdl_ref == ojzdl native on %d frames" % len(fr))
    qemu = os.environ.get("JZDL_QEMU", os.path.expanduser("~/jzdl-qemu"))
    if not os.path.isdir(os.path.join(qemu, "run")):
        report("libjzdl", "SKIP", "no offline runtime at %s (JZDL_QEMU)" % qemu)
    elif not build("ojzdlrun.mips", oj) and run(["make", "-s", "mips"], cwd=oj).returncode:
        report("libjzdl", "SKIP", "cannot build ojzdlrun.mips (XB)")
    else:
        out = os.path.join(os.path.dirname(fr["black"]), "gate")
        r = run(["sh", os.path.join(oj, "gate.sh"), out, binp], env=dict(os.environ, PY=sys.executable))
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output"
        ok = r.returncode == 0 and "GATE PASS" in r.stdout
        report("libjzdl", "PASS" if ok else "FAIL", "gate.sh: %s" % last)
    return golden


# ------------------------------------------------------------------ T41
def t41(mdir, binp, fr):
    golden = {}
    on = os.path.join(REPO, "tools", "onna")
    if not build("onnaq", on):
        report("onnaq", "SKIP", "cannot build tools/onna/onnaq")
        return golden
    tmpd = os.path.dirname(fr["black"])
    bad, nheads = [], 0
    for name, f in fr.items():
        pre = os.path.join(tmpd, "t41_" + name)
        r = run([os.path.join(on, "onnaq"), "-e", "cpu", binp, "-", f, pre])
        heads = sorted(p for p in os.listdir(tmpd) if p.startswith("t41_%s." % name) and p.endswith(".f32"))
        if r.returncode or not heads:
            bad.append("%s (%s)" % (name, r.stdout.strip().splitlines()[-1:] if r.stdout else "no output"))
            continue
        nheads = len(heads)
        golden[name] = [md5(os.path.join(tmpd, h)) for h in heads]
    report("onnaq", "FAIL" if bad else "PASS", ("failed on " + ", ".join(bad)) if bad else
           "graph from the .bin, %d heads, %d frames" % (nheads, len(fr)))
    # the NNA driver against the host oracle, every conv checked
    shutil.copy(os.path.join(on, "onnaq"), os.path.join(tmpd, "onnaq.cpu"))
    if run(["make", "-s", "ORACLE=1", "onnaq"], cwd=on).returncode:
        report("oracle", "SKIP", "ORACLE=1 build failed (needs a QEMU checkout for oracle/mkoracle.sh)")
    else:
        f = fr["noise"]
        pre = os.path.join(tmpd, "or")
        r = run([os.path.join(on, "onnaq"), "-e", "oracle", "-c", binp, "-", f, pre])
        same = r.returncode == 0 and all(
            md5(os.path.join(tmpd, "or.%d.f32" % k)) == golden["noise"][k] for k in range(len(golden.get("noise", []))))
        report("oracle", "PASS" if same else "FAIL", "-e oracle -c on 'noise': %s" % (
            "every conv == CPU, heads == -e cpu" if same else (r.stdout.strip().splitlines()[-1:] or "differs")))
        run(["make", "-s", "onnaq"], cwd=on)      # back to the CPU build
    onnx, mgk = os.path.join(mdir, "model.onnx"), os.path.join(mdir, "model_quant.mgk")
    mgkdump = os.environ.get("MGKDUMP", os.path.expanduser("~/nnaopen/m416/pc/mgkdump"))
    if not (os.path.exists(onnx) and os.path.exists(mgk)):
        report("executor", "SKIP", "needs model.onnx + model_quant.mgk next to model.bin")
    elif not shutil.which("docker") or not os.path.exists(mgkdump):
        report("executor", "SKIP", "needs Docker + mgkdump (MGKDUMP)")
    else:
        bad = []
        for name, f in fr.items():
            r = run(["sh", os.path.join(REPO, "tools", "nnaop", "m416", "mgkcmp.sh"), onnx, binp, mgk, f, "-"],
                    env=dict(os.environ, ONNAQ=os.path.join(on, "onnaq"), PY=sys.executable))
            if r.returncode:
                bad.append(name)
        report("executor", "FAIL" if bad else "PASS", ("differs on " + ", ".join(bad)) if bad else
               "onnaq == Ingenic's PC executor, every head, %d frames" % len(fr))
    return golden


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--frames", action="append", help="extra 416x320 RGB frames (*.rgb)")
    ap.add_argument("--update-card", action="store_true", help="write the golden md5s into model.json")
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    mdir = os.path.abspath(a.model_dir)
    card, cardp = card_of(mdir)
    tmp = tempfile.mkdtemp(prefix="conform-")
    try:
        fr = frames(a.frames, tmp)
        t31bin = os.path.join(mdir, "model_t31.bin")
        t41bin = os.path.join(mdir, "model_t41.bin")
        if card:
            soc = card.get("soc", [""])[0]
            t31bin = os.path.join(mdir, card["file"]) if soc == "T31" else None
            t41bin = os.path.join(mdir, card["file"]) if soc == "T41" else None
        gold = {}
        if t31bin and os.path.exists(t31bin):
            print("== T31 %s (md5 %s)" % (os.path.relpath(t31bin), md5(t31bin)[:8]))
            gold["t31_output_md5"] = t31(mdir, t31bin, fr, card)
        if t41bin and os.path.exists(t41bin):
            print("== T41 %s (md5 %s)" % (os.path.relpath(t41bin), md5(t41bin)[:8]))
            gold["t41_heads_md5"] = t41(mdir, t41bin, fr)
        if not gold:
            raise SystemExit("no model.bin / model_t31.bin / model_t41.bin in %s" % mdir)
        failed = [r for r in results if r["status"] == "FAIL"]
        if a.update_card and cardp and not failed:
            card["conformance"] = dict(frames="tools/conform.py synthetic frames (black, white, checker, gradient, noise seed 0)",
                                       checks={r["check"]: r["status"] for r in results}, **gold)
            json.dump(card, open(cardp, "w"), indent=1)
            open(cardp, "a").write("\n")
            print("model.json: conformance written")
        if a.json:
            json.dump(dict(results=results, golden=gold), open(a.json, "w"), indent=1)
        print("RESULT: %s (%d pass, %d fail, %d skip)" % ("FAIL" if failed else "PASS",
              sum(r["status"] == "PASS" for r in results), len(failed), sum(r["status"] == "SKIP" for r in results)))
        sys.exit(1 if failed else 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
