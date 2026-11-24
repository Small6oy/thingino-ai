# onna: open runtime for the T41 NNA (venus-compatible), after tools/ojzdl's layout.
#   make            host build (x86/arm): test_cpu against golden.py's dumps
#   make mips       T41 build with the xb v4 toolchain (on the build host)
#   make check      host test;  make check-mips: MIPS test under user-mode QEMU (-cpu XBurst2)
#   make check-vrq  the MXU3 SiLU requant vs the scalar path, under QEMU -cpu XBurst2-MXU3.1 (MODEL=.bin)
#   make check-vadd the MXU3 add + concat rescale vs the scalar path, every input (MODEL, GRAPH)
CFLAGS ?= -O2
WARN = -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror -Wshadow -Wstrict-prototypes -Wformat=2
# float32 requant/head must round like venus: no fused multiply-add
FP = -ffp-contract=off
XB ?= $(HOME)/xb/tf-v4/output/xb
MIPSCC ?= $(XB)/host/bin/mipsel-linux-gcc
SYSROOT ?= $(XB)/host/mipsel-buildroot-linux-uclibc/sysroot
QEMU ?= $(HOME)/jzdl-qemu/fbuild/qemu-mipsel
GOLD ?= ../gold
MODEL ?= ../yolov5n_t41.bin
GRAPH ?= ../yolov5n.graph

all: test_cpu onnarun

RUN = onnarun.c nnabin.c nnacpu.c nnaeng.c
RUNH = nnabin.h nnacpu.h nnaeng.h
# ORACLE=1: link the host test oracle (QEMU's NNA engine code, built by oracle/mkoracle.sh from
# QEMU_NNA_C) so that `onnarun -e oracle` runs every conv through the NNA driver
QEMU_NNA_C ?= $(HOME)/jzdl-qemu/fork/hw/misc/ingenic-nna.c
ifeq ($(ORACLE),1)
RUN += be_oracle.c oracle/nnaoracle.o
CFLAGS += -DONNA_ORACLE
oracle/nnaoracle.o: oracle/mkoracle.sh oracle/prelude.h oracle/epilogue.c
	oracle/mkoracle.sh $(QEMU_NNA_C) oracle
	$(CC) -O2 -std=gnu11 -w -c oracle/nnaoracle.c -o $@
onnarun: oracle/nnaoracle.o
endif

onnarun: $(RUN) $(RUNH)
	$(CC) $(CFLAGS) $(WARN) $(FP) -o $@ $(RUN) -lm

onnarun.mips: $(RUN) $(RUNH) be_mxu3.c be_mxu3.h mxu3.h
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DONNA_MXU3 -o $@ $(RUN) be_mxu3.c -lm

mxu3test.mips: mxu3test.c mxu3.h
	$(MIPSCC) -O2 $(WARN) -march=mips32r2 -o $@ mxu3test.c

test_cpu: test_cpu.c nnacpu.c nnacpu.h
	$(CC) $(CFLAGS) $(WARN) $(FP) -o $@ test_cpu.c nnacpu.c -lm

vrqtest.mips: vrqtest.c nnabin.c nnacpu.c nnabin.h nnacpu.h mxu3.h
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DONNA_MXU3 -static -o $@ vrqtest.c nnabin.c nnacpu.c -lm

vaddtest.mips: vaddtest.c nnabin.c nnacpu.c nnabin.h nnacpu.h mxu3.h
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DONNA_MXU3 -static -o $@ vaddtest.c nnabin.c nnacpu.c -lm

vrqbench.mips: vrqbench.c nnacpu.c nnacpu.h mxu3.h
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DONNA_MXU3 -static -o $@ vrqbench.c nnacpu.c -lm

rqbench.mips: rqbench.c nnacpu.c nnacpu.h mxu3.h
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DONNA_MXU3 -static -o $@ rqbench.c nnacpu.c -lm

mips: test_cpu.mips onnarun.mips mxu3test.mips vrqtest.mips vaddtest.mips

test_cpu.mips: test_cpu.c nnacpu.c nnacpu.h
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -o $@ test_cpu.c nnacpu.c -lm

check: test_cpu
	./test_cpu $(GOLD) | tail -1

check-mips: test_cpu.mips onnarun.mips mxu3test.mips
	$(QEMU) -cpu XBurst2 -L $(SYSROOT) ./test_cpu.mips $(GOLD) | tail -1

check-vrq: vrqtest.mips
	$(QEMU) -cpu XBurst2-MXU3.1 ./vrqtest.mips $(MODEL) 2000 | tail -1

check-vadd: vaddtest.mips
	$(QEMU) -cpu XBurst2-MXU3.1 ./vaddtest.mips $(MODEL) $(GRAPH) | tail -1

clean:
	rm -f test_cpu test_cpu.mips onnarun onnarun.mips mxu3test.mips vrqtest.mips vaddtest.mips vrqbench.mips rqbench.mips oracle/nnaoracle.c oracle/nnaoracle.o

.PHONY: all mips check check-mips check-vrq check-vadd clean
