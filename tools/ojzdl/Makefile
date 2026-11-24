# ojzdl: open JZDL runtime.
#   make            host build (x86/arm, for the reference check)
#   make mips       T31 build with the xb v3 toolchain (on the build host)
CFLAGS ?= -O2
WARN = -std=c99 -D_POSIX_C_SOURCE=200112L -Wall -Wextra -Werror -Wshadow -Wstrict-prototypes -Wformat=2
# no fused multiply-add: the head's float32(acc + bias) * scale must round like libjzdl
FP = -ffp-contract=off
XB ?= $(HOME)/xb/tf-v3/output/xb
MIPSCC ?= $(XB)/host/bin/mipsel-linux-gcc

all: ojzdlrun

SRCS = ojzdlrun.c ojzdl.c ojzdl_mxu2.c ojzdl_resize.c
HDRS = ojzdl.h ojzdl_mxu2.h mxu2.h

ojzdlrun: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(WARN) $(FP) -o $@ $(SRCS) -lm

mips: ojzdlrun.mips mxu2test.mips

ojzdlrun.mips: $(SRCS) $(HDRS)
	$(MIPSCC) $(CFLAGS) $(WARN) $(FP) -march=mips32r2 -DOJZDL_MXU2 -o $@ $(SRCS) -lm

mxu2test.mips: mxu2test.c mxu2.h
	$(MIPSCC) -O2 $(WARN) -march=mips32r2 -o $@ mxu2test.c

clean:
	rm -f ojzdlrun ojzdlrun.mips mxu2test.mips

.PHONY: all mips clean
