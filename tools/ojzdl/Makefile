# Copyright (C) 2026 Thingino Project
# SPDX-License-Identifier: MIT
# ojzdl: open JZDL runtime.
#   make            host build (x86/arm, for the reference check)
#   make mips       T31 build with the xb v3 toolchain (on the build host)
CFLAGS ?= -O2
WARN = -std=c99 -D_POSIX_C_SOURCE=200112L -Wall -Wextra -Werror -Wshadow -Wstrict-prototypes -Wformat=2
# no fused multiply-add: the head's float32(acc + bias) * scale must round like libjzdl
FP = -ffp-contract=off
XB ?= $(HOME)/xb/tf-v3/output/xb
MIPSCC ?= $(XB)/host/bin/mipsel-linux-gcc

all: ojzdlrun nv12test

SRCS = ojzdlrun.c ojzdl.c ojzdl_mxu2.c ojzdl_resize.c
HDRS = ojzdl.h ojzdl_mxu2.h mxu2.h

ojzdlrun: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(WARN) $(FP) -o $@ $(SRCS) -lm

nv12test: nv12test.c ojzdl_resize.c $(HDRS)
	$(CC) $(CFLAGS) $(WARN) $(FP) -o $@ nv12test.c ojzdl_resize.c -lm

mips: ojzdlrun.mips mxu2test.mips nv12test.mips

# T41 (XBurst2): the vertical pass on MXU3; runs on core 0 only (nv12test pins itself)
XB4 ?= $(HOME)/xb/tf-v4/output/xb
MIPSCC4 ?= $(XB4)/host/bin/mipsel-linux-gcc
t41: nv12test.t41

nv12test.t41: nv12test.c ojzdl_resize.c $(HDRS)
	$(MIPSCC4) $(CFLAGS) $(WARN) $(FP) -DOJZDL_MXU3 -static -o $@ nv12test.c ojzdl_resize.c -lm

nv12test.mips: nv12test.c ojzdl_resize.c ojzdl_mxu2.c $(HDRS)
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DOJZDL_MXU2 -static -o $@ nv12test.c ojzdl_resize.c ojzdl_mxu2.c -lm

ojzdlrun.mips: $(SRCS) $(HDRS)
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DOJZDL_MXU2 -o $@ $(SRCS) -lm

mxu2test.mips: mxu2test.c mxu2.h
	$(MIPSCC) -O2 $(WARN) -march=mips32r2 -o $@ mxu2test.c

clean:
	rm -f ojzdlrun ojzdlrun.mips mxu2test.mips nv12test nv12test.mips nv12test.t41

.PHONY: all mips t41 clean
