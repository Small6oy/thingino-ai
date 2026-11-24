/* nnabench: cost of each NN instruction on silicon, issued directly (inline .word) and through
 * onna's backend (function pointer + switch). Pins to CPU0. */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "mxu3.h"
#include "be_mxu3.h"

static double now_us(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

#define N 100000
#define BENCH(name, body) do { double t0 = now_us(); for (int i = 0; i < N; i++) { body; } \
	printf("%-34s %7.3f us\n", name, (now_us() - t0) / N); } while (0)

int main(void)
{
	cpu_set_t cs;
	struct nna_be be;
	unsigned char v[64] __attribute__((aligned(64)));

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	nna_be_mxu3(&be);
	memset(v, 0, 64);
	NNCMD(0);
	VWR0_SET(0x33);
	BENCH("NNRWR direct", NNRWR(0x3b));
	BENCH("VWR0_SET + NNRWR direct", { VWR0_SET(0x33); NNRWR(0x3b); });
	BENCH("rwr via backend", be.rwr(be.ctx, 0x3b, 0x33));
	VPR0_LOAD(v);
	BENCH("NNDWR 0x20 direct (F)", NNDWR(0x20));
	BENCH("VPR0_LOAD + NNDWR 0x20 direct", { VPR0_LOAD(v); NNDWR(0x20); });
	BENCH("dwr 0x20 via backend", be.dwr(be.ctx, 0x20, v));
	BENCH("NNDRD 0 direct", NNDRD(0));
	BENCH("NNDRD 0 + VPR0_STORE direct", { NNDRD(0); VPR0_STORE(v); });
	BENCH("drd via backend", be.drd(be.ctx, 0, v));
	VWR0_SET(0);
	BENCH("NNMAC 0 direct (1 tap, 1 plane)", NNMAC(0));
	BENCH("NNCMD 0x83 direct", NNCMD(0x83));
	BENCH("MAC + CPOC + 16 DRD direct", { NNMAC(0); NNCMD(0x83); NNDRD(0); NNDRD(1); NNDRD(2); NNDRD(3);
		NNDRD(4); NNDRD(5); NNDRD(6); NNDRD(7); NNDRD(8); NNDRD(9); NNDRD(10); NNDRD(11);
		NNDRD(12); NNDRD(13); NNDRD(14); NNDRD(15); });
	return 0;
}
