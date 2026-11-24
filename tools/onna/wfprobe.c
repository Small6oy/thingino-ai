/* wfprobe: F writes under a multi-word WFCW program, interleaved with MACs as a tiled conv does:
 * per tile, NPT NNDWR 0x20 (2 pixels each), then MACs at word t*TW and t*TW + 4 (2 rows x 4
 * pixels, pitch 0x39). Written pixel n holds n/32+1 in channel n%32 only, identity weights, so
 * each output pixel's non-zero channel names the written pixel it read. Same code on the oracle
 * and silicon.
 *   env WFCW="0x802,0x80e" (0x59 words, 0x5a = their count), PLANE=0x10 (0x37), PITCH=8 (0x39),
 *       MACTL=0x12 (0x38), NPT=8 (NNDWR per tile), TW=64 (MAC word step per tile), TILES=32,
 *       R5B=0 (0x5b), R21=0 (0x21 and 0x23 when non-zero) 
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
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

static unsigned env(const char *n, unsigned d)
{
	const char *v = getenv(n);

	return v ? (unsigned)strtoul(v, 0, 0) : d;
}

int main(void)
{
	struct nna_be be;
	unsigned plane = env("PLANE", 0x10), pitch = env("PITCH", 8), mactl = env("MACTL", 0x12);
	unsigned npt = env("NPT", 8), tw = env("TW", 64), tiles = env("TILES", 32);
	const char *prog = getenv("WFCW") ? getenv("WFCW") : "0x802,0x80e";
	static uint8_t f[8192 * 32];
	uint8_t ent[256], v[64];
	unsigned words[16], nw = 0;

	for (const char *p = prog; *p && nw < 16;) {
		char *e;

		words[nw++] = (unsigned)strtoul(p, &e, 0);
		p = *e ? e + 1 : e;
	}
#ifdef ONNA_ORACLE
	nna_be_oracle(&be);
#else
	cpu_set_t cs;

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	nna_be_mxu3(&be);
#endif
	setvbuf(stdout, NULL, _IONBF, 0);
	be.cmd(be.ctx, 0);
	if (env("R21", 0)) {
		be.rwr(be.ctx, 0x21, env("R21", 0));
		be.rwr(be.ctx, 0x23, env("R21", 0));
	}
	be.rwr(be.ctx, 0x28, 2);
	be.rwr(be.ctx, 0x3b, 0x33); be.rwr(be.ctx, 0x46, 0); be.rwr(be.ctx, 0x55, 2);
	be.rwr(be.ctx, 0x38, mactl); be.rwr(be.ctx, 0x3f, 0xff); be.rwr(be.ctx, 0x2f, 0);
	be.rwr(be.ctx, 0x2e, 0x7ff); be.rwr(be.ctx, 0x2c, 0x7ff);
	be.rwr(be.ctx, 0x3e, 1); be.rwr(be.ctx, 0x3c, 0); be.rwr(be.ctx, 0x3a, 0);
	be.rwr(be.ctx, 0x39, pitch); be.rwr(be.ctx, 0x37, plane);
	be.rwr(be.ctx, 0x5c, 0);
	for (int s = 0; s < 4; s++) {		/* identity weights, msf 0: oc j <- ic j */
		memset(ent, 0, sizeof(ent));
		if (s == 0)
			for (int o = 0; o < 32; o++)
				ent[o * 8 + o / 4] = (uint8_t)(1u << (2 * (o % 4)));
		be.dwr(be.ctx, 0x00, ent); be.dwr(be.ctx, 0x02, ent + 64);
		be.dwr(be.ctx, 0x04, ent + 128); be.dwr(be.ctx, 0x07, ent + 192);
	}
	be.rwr(be.ctx, 0x5b, env("R5B", 0)); be.rwr(be.ctx, 0x45, 2);
	printf("weights in, program %s\n", prog);
	/* clear pass with the one-word program, all F RAM to zero */
	be.rwr(be.ctx, 0x58, 0); be.rwr(be.ctx, 0x59, 0x801); be.rwr(be.ctx, 0x5a, 1);
	be.cmd(be.ctx, 0x60);
	memset(f, 0, sizeof(f));
	be.rwr(be.ctx, 0x50, 0);
	for (int i = 0; i < 4096; i += 2)
		be.dwr(be.ctx, 0x20, f);
	/* the program under test */
	be.rwr(be.ctx, 0x58, 0);
	for (unsigned i = 0; i < nw; i++)
		be.rwr(be.ctx, 0x59, words[i]);
	be.rwr(be.ctx, 0x5a, nw);
	be.cmd(be.ctx, 0x60);
	printf("F cleared, program set\n");
	for (int a = 0; a < 8192; a++)
		f[a * 32 + a % 32] = (uint8_t)(a / 32 + 1);
	be.rwr(be.ctx, 0x50, 0);
	for (unsigned t = 0; t < tiles; t++) {
		for (unsigned i = 0; i < npt; i++)
			be.dwr(be.ctx, 0x20, f + 64L * (t * npt + i));
		for (unsigned m = 0; m < 2; m++) {
			be.rwr(be.ctx, 0x29, 0); be.rwr(be.ctx, 0x2b, 0);
			be.rwr(be.ctx, 0x31, 0); be.rwr(be.ctx, 0x30, 0);
			be.mac(be.ctx, 0, t * tw + 4 * m);
			be.cmd(be.ctx, 0x83);
			printf("tile %2u mac %u word %4u reads:", t, m, t * tw + 4 * m);
			for (int p = 0; p < 8; p++) {
				int32_t s[32];
				int n = 0;

				be.drd(be.ctx, 2 * p, v);
				memcpy(s, v, 64);
				be.drd(be.ctx, 2 * p + 1, v);
				memcpy(s + 16, v, 64);
				printf(" ");
				for (int j = 0; j < 32; j++)
					if (s[j])
						printf("%s%d", n++ ? "+" : "", (s[j] - 1) * 32 + j);
				if (!n)
					printf("-");
			}
			printf("\n");
		}
	}
	return 0;
}
