/* fprobe: which F RAM address does each output pixel of a MAC read? Identity weights, and pixel
 * address a holds a/32+1 in channel a%32 only (all F RAM written), so each output pixel's
 * non-zero channel names the address it read, per tap. Same code on the oracle and silicon.
 *   fprobe [mactl pitch plane word fy fx taps]   (defaults: the generic 1x1 setup) */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnaeng.h"
#ifdef ONNA_ORACLE
#include "be_oracle.h"
#else
#include "be_mxu3.h"
#endif

int main(int argc, char **argv)
{
	struct nna_be be;
	unsigned mactl = argc > 1 ? strtoul(argv[1], 0, 0) : 0x12, pitch = argc > 2 ? strtoul(argv[2], 0, 0) : 20;
	unsigned plane = argc > 3 ? strtoul(argv[3], 0, 0) : 12, word = argc > 4 ? strtoul(argv[4], 0, 0) : 0;
	unsigned fy = argc > 5 ? strtoul(argv[5], 0, 0) : 0, fx = argc > 6 ? strtoul(argv[6], 0, 0) : 0;
	unsigned taps = argc > 7 ? strtoul(argv[7], 0, 0) : 1;
	static uint8_t f[4096 * 32];
	uint8_t ent[256], v[64];

#ifdef ONNA_ORACLE
	nna_be_oracle(&be);
#else
	cpu_set_t cs;

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	nna_be_mxu3(&be);
#endif
	be.cmd(be.ctx, 0);
	be.rwr(be.ctx, 0x3b, 0x33); be.rwr(be.ctx, 0x46, 0); be.rwr(be.ctx, 0x55, 2);
	be.rwr(be.ctx, 0x38, mactl); be.rwr(be.ctx, 0x3f, 0xff); be.rwr(be.ctx, 0x2f, 0);
	be.rwr(be.ctx, 0x2e, 0x7ff); be.rwr(be.ctx, 0x2c, 0x7ff); be.rwr(be.ctx, 0x28, 2);
	be.rwr(be.ctx, 0x3e, taps); be.rwr(be.ctx, 0x3c, 0); be.rwr(be.ctx, 0x3a, 0);
	be.rwr(be.ctx, 0x39, pitch); be.rwr(be.ctx, 0x37, plane);
	/* identity weights, msf 0 (w = v): oc j <- ic j weight 1, for every tap group */
	be.rwr(be.ctx, 0x5c, 0);
	for (int g = 0; g < __builtin_popcount(taps); g++)
		for (int s = 0; s < 4; s++) {
			memset(ent, 0, sizeof(ent));
			if (s == 0)
				for (int o = 0; o < 32; o++)
					ent[o * 8 + o / 4] = (uint8_t)(1u << (2 * (o % 4)));
			be.dwr(be.ctx, 0x00, ent); be.dwr(be.ctx, 0x02, ent + 64);
			be.dwr(be.ctx, 0x04, ent + 128); be.dwr(be.ctx, 0x07, ent + 192);
		}
	be.rwr(be.ctx, 0x5b, 0); be.rwr(be.ctx, 0x45, 2); be.rwr(be.ctx, 0x58, 0);
	be.rwr(be.ctx, 0x59, 0x801); be.rwr(be.ctx, 0x5a, 1);
	be.cmd(be.ctx, 0x60);
	memset(f, 0, sizeof(f));
	for (int a = 0; a < 4096; a++)
		f[a * 32 + a % 32] = (uint8_t)(a / 32 + 1);
	be.rwr(be.ctx, 0x50, 0);
	for (int i = 0; i < 4096; i += 2)
		be.dwr(be.ctx, 0x20, f + 32L * i);
	be.rwr(be.ctx, 0x29, fy); be.rwr(be.ctx, 0x2b, fx); be.rwr(be.ctx, 0x31, 0); be.rwr(be.ctx, 0x30, 0);
	be.mac(be.ctx, 1, word);
	be.cmd(be.ctx, 0x83);
	for (int p = 0; p < 8; p++) {
		int32_t s[32];

		be.drd(be.ctx, 2 * p, v);
		memcpy(s, v, 64);
		be.drd(be.ctx, 2 * p + 1, v);
		memcpy(s + 16, v, 64);
		printf("px %d reads:", p);
		for (int j = 0; j < 32; j++)
			if (s[j])
				printf(" %d", (s[j] - 1) * 32 + j);
		printf("\n");
	}
	return 0;
}
