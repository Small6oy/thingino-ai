# nna: T41 NNA (venus) tools

Run models on the T41's NNA with Ingenic's Magik venus runtime (0.1.2.5, the one that matches
magik-transform-tools 2.0.4).

| File | What |
|---|---|
| `nnatest.cpp` | runs a Magik/venus YOLOv5 (3 heads, strides 8/16/32) on an NV12 frame: letterbox on the AIP, timings, boxes. Based on Ingenic's `inference_nv12.cpp` sample. |
| `nnappp.cpp` | runs the thingino-ai person/pet/package detector (2 heads, strides 32/16, persondet anchors) on pre-resized 416x320 RGB frames: timings, boxes, and with `NNAPPP_DUMP=dir` the raw head outputs, so they can be compared with Ingenic's PC executor (`train/t41/mgkrun`). |
| `build.sh` | cross-builds both with the T41 toolchain in Docker; needs `MAGIK` (magik-toolkit checkout) and `TC_IMAGE`. Output in `out/toolkit/` (binaries + the vendor libs, never committed). |
| `nnprobe.c` | which MXU3 / NN instructions each T41 core accepts (OK or SIGILL per instruction, CPU0 and CPU1). Showed that the NN instructions run on CPU0 only. Plain C, any T41 toolchain: `mipsel-linux-gcc -O2 -o nnprobe nnprobe.c`. |
| `nnatrace.c` | `LD_PRELOAD` logger of what venus/libdrivers ask of the kernel: opens of `/dev` and `/proc` files, mmap/munmap, ioctls with their argument bytes. `mipsel-linux-gcc -shared -fPIC -O2 -o nnatrace.so nnatrace.c -ldl`. |
| `pincpu.c` | `pincpu CPU cmd [args...]`: run a command pinned to one CPU (the camera's busybox has no `taskset`). |
| `t41l2/` | kernel module that shrinks the L2 cache to 128 KiB at runtime, so the NNA's ORAM sits where venus expects it. See its README. |

## Running on a T41 camera without flashing

The NNA needs memory reserved at boot (`nmem`) and the L2/ORAM split. Both can be done on a stock
Thingino image:

```
fw_setenv rmem '44M@0x4000000 nmem=20M@0x6c00000'     # the stock bootcmd passes rmem=${rmem}
reboot
insmod t41l2.ko                                        # L2 256 -> 128 KiB, ORAM at 0x12620000
LD_LIBRARY_PATH=/tmp/nna/lib ./nnappp model.bin 20 0.25 frame1.rgb frame2.rgb
fw_setenv rmem 64M@0x4000000; reboot                   # back to stock
```

- venus's NN instructions only run on **CPU0**; both tools pin themselves there.
- Keep `/tmp` small: it is RAM. Ship one model and a few frames, and delete them after the run.
- Measured with the person/pet/package v1 model on a Wyze Cam v4: ~37 ms per frame, ~3.9 MB of
  venus memory (so `nmem` 8 MiB should be enough), output bit-identical to the PC executor.
