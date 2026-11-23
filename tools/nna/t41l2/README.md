# t41l2: shrink the T41 L2 at runtime (no boot loader flash)

A one-shot kernel module that does what U-Boot's `CONFIG_T41_L2_128K` (thingino branch `t41-nna-oram`) does, but from a running kernel. It sets CCU MSCR to 128 KiB of L2, so the NNA gets its 384 KiB of ORAM at 0x12620000, which is where the vendor soc-nna driver and venus expect it. A reboot undoes it.

How it does the switch:
- Everything runs under `stop_machine()`.
- Each CPU writes back its L1.
- The master CPU then writes back the whole L2 and writes MSCR. Nothing is stored between those two steps; the disassembly was checked.
- The CPUs hand off through uncached (KSEG1) flags.

Build on a Linux host, against the matching kernel tree. The vermagic must match the camera (4.4.94 SMP preempt):

    B=/workspace/output/build/<stamp>/wyze_cam4_t41nq_os04d10_atbm6062s-4.4.94-uclibc
    docker run --rm -u $(id -u):$(id -g) -v ~/tf-wyze_cam4_t41nq_os04d10_atbm6062s:/workspace:ro -v ~/t41l2:/mod \
      ghcr.io/themactep/thingino-builder-image:latest bash -c \
      "export PATH=$B/host/bin:\$PATH; make -C $B/build/linux-c6bc* M=/mod ARCH=mips CROSS_COMPILE=mipsel-linux- modules"

Use: `insmod t41l2.ko` (`share=1` is the default), then `dmesg | grep t41l2`.

nmem without editing bootcmd: the stock bootcmd passes `rmem=${rmem}`, so
`fw_setenv rmem '44M@0x4000000 nmem=20M@0x6c00000'` puts both on the kernel cmdline.
To revert: `fw_setenv rmem 64M@0x4000000`.

Emulator 2026-11-18, stock 2243 U-Boot image + the rmem trick:
- MSCR went 0x02000830 → 0x02000430.
- yolov5n loaded and ran, with no BUG.
- rvd kept streaming 2560×1440.
