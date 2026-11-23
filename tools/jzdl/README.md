# jzdl tools: read, write and load-check JZDL models (T31)

Format notes: [docs/jzdl-format.md](../../docs/jzdl-format.md).

| File | What it does |
|---|---|
| `jzdl.py` | Parser + writer. `Model.parse(bytes)` → layers (type, bottoms, tops, int32 params, data blobs). `Model.to_bytes()` writes it back. CLI: `python3 jzdl.py MODEL.bin` prints the layer table and checks the round trip. Pure Python, no dependencies. |
| `jzdl_loadcheck.py` | Runs **libjzdl.m.so's own loader** on a `.bin` under Unicorn on the Mac (no camera). OK = `load_model` returned 0 and read the whole file. `--trace t.json` dumps every `fread` (offset, size, layer). Only loading is emulated: the forward pass uses MXU1, which Unicorn lacks. |

## Setup (once)

```sh
python3 -m venv ~/.venvs/jzdl && ~/.venvs/jzdl/bin/pip install unicorn capstone pyelftools
```

## Checks (2026-11-23)

```sh
M=../aitest/out/v3
python3 jzdl.py $M/magik_model_persondet.bin      # roundtrip byte-identical: True
python3 jzdl.py $M/magik_model_yolov5.bin         # roundtrip byte-identical: True
~/.venvs/jzdl/bin/python -I jzdl_loadcheck.py $M/libjzdl.m.so $M/magik_model_persondet.bin   # OK
```

- Both sample models round-trip byte for byte.
- The parser's 100 (persondet) and 353 (yolov5) blob offsets match the emulated runtime's freads exactly.
- A persondet copy with one head bias changed (+1000) loads OK.
- A truncated file fails (`load_model -> -1`).

## Editing example

```python
import struct, jzdl
m = jzdl.Model.parse(open('magik_model_persondet.bin', 'rb').read())
head = m.layers[26]                                # t51, stride-16 head
b = list(struct.unpack('<18i', head.blobs[1]))     # int32 bias
b[5] += 1000                                       # person logit, anchor 0
head.blobs[1] = struct.pack('<18i', *b)
open('edit.bin', 'wb').write(m.to_bytes())
```

## Step 2 tools (2026-11-23)

| File | What it does |
|---|---|
| `jzdl_ref.py` | **Bit-exact NumPy forward pass** for version-5 models with the persondet layer set. It matched the runtime on every blob and the packed output on two frames. `forward(model, rgb)` returns all blobs. |
| `probe.py` | `probe.py MODEL.bin BLOB OUT.bin` writes a model whose output is the int8 blob BLOB, through an identity type-51 head, since `run()` only returns float blobs. |
| `jzdlrun.cpp` | Runs a model on one raw RGB frame and dumps float outputs (`OUT.<blob>.f32` = int32 w, h, c + floats). Build it with the v3 toolchain. It runs under QEMU or on the v3. |
| `experiments.py` | Harness for designed experiments: edit a layer, batch-run it on the build host (`JZDL_BUILD_HOST`) under QEMU, read the probed blobs. |

The offline runtime lives in `~/jzdl-qemu` on a Linux build host (set `JZDL_BUILD_HOST`); the recipe is in jzdl-format.md "Offline runtime". `jzdlrun` build command (on the build host, with a Thingino T31 Buildroot output in `~/xb/tf-v3`):

```sh
B=~/xb/tf-v3/output/xb; J=$(ls -d $B/build/thingino-raptor-hal-*/ingenic-headers/Txx/jzdl | head -1)
$B/host/bin/mipsel-linux-g++ -std=c++11 -O2 -DJZ_MXU=0 -I$J -o jzdlrun jzdlrun.cpp -L. -l:libjzdl.m.so -Wl,-rpath,/tmp/ai:.
```

Two things to know about `jzdlrun`:

- **One blob per call.** Asking for two blob ids in one call (`jzdlrun m.bin in.rgb out 22 28`)
  crashes (seen under QEMU): a second `run()` on the same input isn't safe. Call it once per blob,
  or with no blob for the default packed output (`-999`).
- **On a T31 camera it needs `libstdc++.so.6`** next to it (e.g. in `/tmp/ai`): the image has none.
  `../aitest/build.sh` produces a stripped copy. Like all Ingenic/toolchain libraries, it is never
  committed here.

The QAT training graph and the exporter that used to live in `qat/` are now in
[train/qat/](../../train/qat/).
