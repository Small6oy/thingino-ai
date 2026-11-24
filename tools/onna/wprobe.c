/* wprobe: how many W entries does the NNA hold? Loads G weight groups (4 entries each, 8-bit,
 * msf 2): group g puts (g % 100) + 1 on oc 0 and g / 100 + 1 on oc 1 (input channel 0 only).
 * F is one pixel row with channel 0 = 1. G MACs in a row, each stepping the W pointer (immediate
 * bit 0) onto the next group; each read-back names the group the MAC used.
 *   wprobe [G]   (default 256 = 1024 entries) */
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
	int G = argc > 1 ? atoi(argv[1]) : 256, first_bad = -1;
	struct nna_be be;
	uint8_t ent[256], f[64] __attribute__((aligned(64)));

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
	be.rwr(be.ctx, 0x3b, 0x33); be.rwr(be.ctx, 0x46, 2); be.rwr(be.ctx, 0x28, 2); be.rwr(be.ctx, 0x55, 2);
	be.rwr(be.ctx, 0x38, 0x12); be.rwr(be.ctx, 0x3f, 0xff); be.rwr(be.ctx, 0x2f, 0);
	be.rwr(be.ctx, 0x2e, 0x7ff); be.rwr(be.ctx, 0x2c, 0x7ff); be.rwr(be.ctx, 0x3e, 1);
	be.rwr(be.ctx, 0x3c, 0); be.rwr(be.ctx, 0x3a, 0); be.rwr(be.ctx, 0x39, 4); be.rwr(be.ctx, 0x37, 8);
	be.rwr(be.ctx, 0x5c, 0);
	for (int g = 0; g < G; g++) {
		int v0 = (g % 100) + 1 + 128, v1 = g / 100 + 1 + 128;	/* msf 2: w = v - 128 */

		for (int s = 0; s < 4; s++) {
			memset(ent, 0, sizeof(ent));
			for (int o = 0; o < 32; o++)			/* every other weight v = 128 (0) */
				for (int j = 0; j < 8; j++)
					ent[o * 8 + j] = (uint8_t)((2 << 0) | (2 << 2) | (2 << 4) | (2 << 6)) * (s == 3);
			/* slices of 128 = 0b10000000: only slice 3 is 2 (done above); now oc 0/1, ic 0 */
			for (int o = 0; o < 2; o++) {
				int v = o ? v1 : v0;

				ent[o * 8] = (uint8_t)((ent[o * 8] & ~3) | ((v >> (2 * s)) & 3));
			}
			be.dwr(be.ctx, 0x00, ent); be.dwr(be.ctx, 0x02, ent + 64);
			be.dwr(be.ctx, 0x04, ent + 128); be.dwr(be.ctx, 0x07, ent + 192);
		}
	}
	be.rwr(be.ctx, 0x5b, 0); be.rwr(be.ctx, 0x45, 2); be.rwr(be.ctx, 0x58, 0);
	be.rwr(be.ctx, 0x59, 0x801); be.rwr(be.ctx, 0x5a, 1); be.cmd(be.ctx, 0x60);
	memset(f, 0, sizeof(f));
	f[0] = 1, f[32] = 1;						/* pixels 0 and 1: channel 0 = 1 */
	be.rwr(be.ctx, 0x50, 0);
	for (int i = 0; i < 4; i++)
		be.dwr(be.ctx, 0x20, f);					/* 8 pixels: rows 0 and 1 */
	be.rwr(be.ctx, 0x29, 0); be.rwr(be.ctx, 0x2b, 0); be.rwr(be.ctx, 0x31, 0); be.rwr(be.ctx, 0x30, 0);
	for (int g = 0; g < G; g++) {
		int32_t s[16];
		uint8_t v[64];

		be.mac(be.ctx, 1, 0);
		be.cmd(be.ctx, 0x83);
		be.drd(be.ctx, 0, v);			/* pixel 0, oc 0..15 */
		memcpy(s, v, 64);
		for (int k = 1; k < 16; k++)
			be.drd(be.ctx, (unsigned)k, v);
		int got = (s[1] - 1) * 100 + (s[0] - 1);
		if (got != g && first_bad < 0) {
			first_bad = g;
			printf("group %d (entry %d): MAC used group %d (sums %d %d)\n", g, 4 * g, got, s[0], s[1]);
		}
	}
	if (first_bad < 0)
		printf("all %d groups (%d entries) read back in order\n", G, 4 * G);
	return 0;
}
