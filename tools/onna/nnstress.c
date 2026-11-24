/* nnstress (stability test only): the NN unit alone, no MXU3 loads/stores in the loop. One-word F
 * program, identity weights; per iteration 8 x NNDWR 0x20 from VPR 0, a MAC, CPOC|CLROC, 16 NNDRD
 * (into VPR 0). Pinned to CPU0.   nnstress [seconds] [qsum] [mem]
 *   qsum: BT + 0x84/0x8b + 4 NNDRD 0x20; mem: each F vector loaded from a 1 MB buffer (LAO) and each
 *   read-back stored to another (SAO), as a runtime does; mem 2: plus `sync` after each tile */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nnaeng.h"
#include "be_mxu3.h"
#include "mxu3.h"

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
	int secs = argc > 1 ? atoi(argv[1]) : 60, qsum = argc > 2 && atoi(argv[2]), mem = argc > 3 ? atoi(argv[3]) : 0;
	uint8_t *src, *dst;
	size_t so = 0, dO = 0;
	struct nna_be be;
	cpu_set_t cs;
	static uint8_t ent[256] __attribute__((aligned(64))), f[64] __attribute__((aligned(64)));
	static int32_t bt[64] __attribute__((aligned(64)));

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	setvbuf(stdout, NULL, _IONBF, 0);
	nna_be_mxu3(&be);
	be.cmd(be.ctx, 0);
	be.rwr(be.ctx, 0x28, 2); be.rwr(be.ctx, 0x3b, 0x33); be.rwr(be.ctx, 0x46, 0); be.rwr(be.ctx, 0x55, 2);
	be.rwr(be.ctx, 0x38, 0x12); be.rwr(be.ctx, 0x3f, 0xff); be.rwr(be.ctx, 0x2f, 0);
	be.rwr(be.ctx, 0x2e, 0x7ff); be.rwr(be.ctx, 0x2c, 0x7ff); be.rwr(be.ctx, 0x3e, 1);
	be.rwr(be.ctx, 0x3c, 0); be.rwr(be.ctx, 0x3a, 0); be.rwr(be.ctx, 0x39, 8); be.rwr(be.ctx, 0x37, 16);
	be.rwr(be.ctx, 0x43, 0); be.rwr(be.ctx, 0x44, 3);
	be.rwr(be.ctx, 0x5c, 0);
	for (int s = 0; s < 4; s++) {
		memset(ent, 0, sizeof(ent));
		if (s == 0)
			for (int o = 0; o < 32; o++)
				ent[o * 8 + o / 4] = (uint8_t)(1u << (2 * (o % 4)));
		be.dwr(be.ctx, 0x00, ent); be.dwr(be.ctx, 0x02, ent + 64);
		be.dwr(be.ctx, 0x04, ent + 128); be.dwr(be.ctx, 0x07, ent + 192);
	}
	be.rwr(be.ctx, 0x5b, 0); be.rwr(be.ctx, 0x45, 2);
	be.rwr(be.ctx, 0x58, 0); be.rwr(be.ctx, 0x59, 0x801); be.rwr(be.ctx, 0x5a, 1);
	be.cmd(be.ctx, 0x60);
	for (int i = 0; i < 32; i++)
		bt[i] = 0, bt[32 + i] = 64;
	be.rwr(be.ctx, 0x5e, 0);
	for (int i = 0; i < 4; i++)
		be.dwr(be.ctx, 0x40, (const uint8_t *)bt + 64 * i);
	memset(f, 3, sizeof(f));
	VPR0_LOAD(f);
	if (posix_memalign((void **)&src, 64, 1 << 20) || posix_memalign((void **)&dst, 64, 1 << 20))
		return 1;
	memset(src, 3, 1 << 20);
	memset(dst, 0, 1 << 20);
	double t0 = now_s(), tl = t0;
	long long it = 0;

	for (;;) {
		VWR0_SET(0);
		NNRWR(0x50);
		for (int i = 0; i < 8; i++) {
			if (mem) {
				VPR0_LOAD(src + so);
				so = (so + 64) & ((1 << 20) - 1);
			}
			NNDWR(0x20);
		}
		VWR0_SET(0);
		NNRWR(0x29); NNRWR(0x2b); NNRWR(0x31); NNRWR(0x30);
		NNMAC(0);
		if (qsum) {
			VWR0_SET(0);
			NNRWR(0x42);
			NNCMD(0x84);
			NNCMD(0x8b);
			for (int k = 0; k < 4; k++) {
				NNDRD(0x20);
				if (mem) {
					VPR0_STORE(dst + dO);
					dO = (dO + 64) & ((1 << 20) - 1);
				}
			}
		} else {
			NNCMD(0x83);
#define RD(k) do { NNDRD(k); if (mem) { VPR0_STORE(dst + dO); dO = (dO + 64) & ((1 << 20) - 1); } } while (0)
			RD(0); RD(1); RD(2); RD(3); RD(4); RD(5); RD(6); RD(7);
			RD(8); RD(9); RD(10); RD(11); RD(12); RD(13); RD(14); RD(15);
		}
		if (mem == 2)
			__asm__ volatile("sync" ::: "memory");
		it++;
		if ((it & 1023) == 0) {
			double t = now_s();

			if (t - tl >= 1) {
				printf("%.0f s: %lld iterations\n", t - t0, it);
				tl = t;
			}
			if (t - t0 >= secs)
				break;
		}
	}
	printf("done %lld iterations\n", it);
	return 0;
}
