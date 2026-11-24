/* rqbench: cost of the fused requant's stages on the camera, per element (test only)
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nnacpu.h"
#include "mxu3.h"

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void)
{
	enum { NV = 200000 };	/* vectors of 16: 3.2M elements */
	static int32_t S[8][16] __attribute__((aligned(64))), tq[8][16] __attribute__((aligned(64)));
	static int32_t B[16] __attribute__((aligned(64)));
	static float Fl[16] __attribute__((aligned(64)));
	uint8_t *lut = malloc(NNA_SILU_LUT), *q = malloc(NV * 16 + 64);
	cpu_set_t cs;
	double t;
	unsigned acc = 0;

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	for (int i = 0; i < NNA_SILU_LUT; i++)
		lut[i] = (uint8_t)(i / 128 + (i > 20000));
	for (int c = 0; c < 16; c++)
		B[c] = c * 7, Fl[c] = 0.37f + c * 0.01f;
	for (int p = 0; p < 8; p++)
		for (int c = 0; c < 16; c++)
			S[p][c] = (p * 977 + c * 131) % 30000 - 3000;
	memset(q, 0, NV * 16 + 64);
	for (int mode = 1; mode < 5; mode += 1 + (mode == 1)) {
		t = now_ms();
		for (int v = 0; v < NV; v += 8) {
			if (mode != 0)
				nna_requant_t(S[0], 8, B, Fl, tq[0]);
			if (mode == 1)
				continue;
			for (int p = 0; p < 8; p++) {
				uint8_t *o = q + (long)(v + p) * 16;

				for (int c = 0; c < 16; c++) {
					int32_t tv = tq[p][c] + 1;

					tv = tv < 0 ? 0 : tv > 32768 ? 32768 : tv;
					o[c] = mode == 4 ? (uint8_t)tv : lut[tv];
				}
			}
		}
		t = now_ms() - t;
		for (int i = 0; i < NV * 16; i += 4096)
			acc += q[i];
		printf("mode %d (%s): %.1f ms, %.1f ns/element\n", mode,
		       (const char *[]){ "-", "MXU3 t only", "-", "MXU3 + full LUT",
					 "MXU3 + clamp/store only" }[mode],
		       t, t * 1e6 / (NV * 16.0));
	}
	printf("(acc %u)\n", acc);
	return 0;
}
