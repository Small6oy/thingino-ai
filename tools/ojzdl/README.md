# ojzdl: open JZDL runtime (T31)

Runs JZDL v5 models (the format of Ingenic's libjzdl 5.4.0) in C99, with output **bit-identical to
libjzdl** and, on a Wyze Cam v3 (T31X), the same speed or slightly faster. MXU2 vector kernels for
every conv type; a plain C path for everything else. Notes, measurements and todo:
[docs/jzdl-runtime.md](../../docs/jzdl-runtime.md). Format: [docs/jzdl-format.md](../../docs/jzdl-format.md).

| File | What it does |
|---|---|
| `ojzdl.h`, `ojzdl.c` | The runtime: load and validate a `.bin`, plan memory (arena + scratch), forward, read the output or any blob, per-layer timing. C path for every layer. |
| `ojzdl_mxu2.c`, `ojzdl_mxu2.h` | MXU2 kernels (T31): 1×1, 1×1 + pool, depthwise, first conv, head, pool, input. Built by `make mips`; a layer outside a kernel's limits uses the C path. |
| `mxu2.h` | MXU2 instruction encodings as `.word` macros (the toolchain has no MXU2 support), with the semantics the kernels rely on. |
| `mxu2test.c` | Self-test of every MXU2 form used, against C models. Must print PASS under QEMU and on the camera. |
| `ojzdlrun.c` | Same CLI and `.f32` output as [`../jzdl/jzdlrun.cpp`](../jzdl/jzdlrun.cpp), plus `-n RUNS` and `-t` (per-layer times). `OJZDL_SIMD=0` forces the C path. |
| `gate.sh` | The bit-exact gate: libjzdl (QEMU) vs ojzdl (QEMU and native) vs `jzdl_ref.py`, every model × frame. Paths via environment, see the script header. |
| `cmpf32.py` | Bitwise compare of two `.f32` dumps. |
| `mxu2dis.py` | Annotates `objdump -d` output with MXU2 mnemonics, using QEMU's `mxu2.decode`. |

```sh
make                     # host build (x86/arm), C path only
make mips                # T31 build with MXU2 + mxu2test.mips (XB=<Thingino T31 Buildroot output>)
./ojzdlrun -n 20 -t MODEL.bin FRAME_416x320.rgb out
./gate.sh OUTDIR MODEL.bin [MODEL.bin ...]
```

```c
const char *err;
ojzdl_net *n = ojzdl_load_file("model.bin", &err);
ojzdl_input_rgb(n, rgb);          /* in_h x in_w x 3 uint8, HWC */
ojzdl_forward(n);
size_t cnt;
const float *out = ojzdl_output(n, &cnt);   /* packed heads, as libjzdl's run(-999) */
ojzdl_free(n);
```

No Ingenic code or binaries are in here. The arithmetic was found by designed experiments
(docs/jzdl-format.md "Arithmetic"); the MXU2 kernels are written from the instruction semantics,
not from libjzdl. The libjzdl library and models are only needed to run the comparison gate.
