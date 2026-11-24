/* rqvtest: is MXU3's vector requant bit-identical to the scalar float32 path?
 *   t = rint(float32(S + B) * Fl): scalar (proven against venus) vs VREQUANT16 (ADDW, FCVT int->f,
 *   FMULW, FCVT f->int). Random cases over the real ranges plus constructed ties. Prints mismatches. 
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mxu3.h"

static long scalar_t(int32_t s, int32_t b, float fl)
{
	float v = (float)(s + b) * fl;
	const float m = 12582912.0f;

	if (v > -4194304.0f && v < 4194304.0f)
		return (long)((v + m) - m);
	return lrintf(v);
}

static uint32_t rng = 12345;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng; }

int main(int argc, char **argv)
{
	long n = argc > 1 ? atol(argv[1]) : 2000000, bad = 0, ties = 0;
	int32_t S[16] __attribute__((aligned(64))), B[16] __attribute__((aligned(64)));
	float Fl[16] __attribute__((aligned(64)));
	int32_t t[16] __attribute__((aligned(64)));

	for (long it = 0; it < n / 16; it++) {
		for (int i = 0; i < 16; i++) {
			S[i] = (int32_t)(rnd() % 2000001) - 1000000;		/* sums seen: +-465k */
			B[i] = (int32_t)(rnd() % 600001) - 300000;		/* biases seen: +-212k */
			Fl[i] = (float)(rnd() % 2000000 + 1) / 1.0e6f;		/* scales seen: 0.04..1.7 */
			if ((it & 7) == 0) {					/* construct ties: (S+B)*Fl = k + 0.5 */
				Fl[i] = 0.5f;
				S[i] = (int32_t)(rnd() % 20001) - 10000;
				B[i] = (int32_t)(rnd() & 1);
				ties++;
			}
		}
		VREQUANT16(S, B, Fl, t);
		for (int i = 0; i < 16; i++) {
			long want = scalar_t(S[i], B[i], Fl[i]);

			if (t[i] != want) {
				if (bad < 5)
					printf("mismatch: S %d B %d Fl %.9g -> vector %d scalar %ld\n", S[i], B[i], Fl[i], t[i], want);
				bad++;
			}
		}
	}
	printf("%ld cases (%ld constructed ties): %ld mismatches -> %s\n", n, ties, bad, bad ? "FAIL" : "PASS");
	return bad != 0;
}
