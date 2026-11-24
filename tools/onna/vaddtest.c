/* vaddtest: nna_vadd and nna_vrescale (MXU3) against the scalar reference for every input: all
 * 65536 (a, b) pairs of each add record of a venus .bin, and all 256 codes for each concat factor
 * the graph uses plus random factors. Camera or user-mode QEMU -cpu XBurst2-MXU3.1. 
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "nnabin.h"
#include "nnacpu.h"

int main(int argc, char **argv)
{
	struct nna_bin b;
	struct nna_graph g;
	static uint8_t A[65536] __attribute__((aligned(64))), Bv[65536] __attribute__((aligned(64)));
	static uint8_t v[65536] __attribute__((aligned(64))), r[65536];
	long cases = 0, bad = 0;
	uint32_t rs = 7;

	if (argc < 3 || nna_bin_load(&b, argv[1]) || nna_graph_load(&g, argv[2], &b))
		return fprintf(stderr, "usage: vaddtest model.bin model.graph\n"), 2;
	for (int i = 0; i < 65536; i++)
		A[i] = (uint8_t)(i >> 8), Bv[i] = (uint8_t)i;
	for (int k = 0; k < b.nadd; k++) {
		const struct nna_add_rec *ad = &b.add[k];

		nna_vadd(A, Bv, 65536 - 7, ad->wa, ad->wb, ad->B, ad->T, v);	/* odd length: tail too */
		nna_add_ref(A, Bv, 65536 - 7, ad->wa, ad->wb, ad->B, ad->T, r);
		for (int i = 0; i < 65536 - 7; i++, cases++)
			if (v[i] != r[i] && bad++ < 5)
				printf("add %d a %d b %d: vec %u ref %u\n", k, A[i], Bv[i], v[i], r[i]);
	}
	/* every concat factor of the graph, then 200 random ones; inputs: all 256 codes, 64 times */
	float fa[64];
	int nf = 0;

	for (int k = 0; k < g.nnode; k++)
		for (int j = 0; j < g.node[k].nin && g.node[k].op == OP_CONCAT; j++)
			if (g.node[k].a[j] != 1.0f && nf < 64)
				fa[nf++] = g.node[k].a[j];
	printf("concat factors in the graph: %d\n", nf);
	for (int k = 0; k < nf + 200; k++) {
		float a = fa[k < nf ? k : 0];

		if (k >= nf) {
			rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
			a = (float)(rs % 100000u) / 25000.0f;
		}
		nna_vrescale(Bv, 256 * 64 - 5, a, v);
		nna_rescale(Bv, 256 * 64 - 5, a, r);
		for (int i = 0; i < 256 * 64 - 5; i++, cases++)
			if (v[i] != r[i] && bad++ < 10)
				printf("rescale %g x %d: vec %u ref %u\n", (double)a, Bv[i], v[i], r[i]);
	}
	printf("vaddtest: %ld cases, %ld mismatches: %s\n", cases, bad, bad ? "FAIL" : "PASS");
	return bad != 0;
}
