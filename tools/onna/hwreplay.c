/* hwreplay: run a traced venus instruction stream (QEMU ingenic-nna.log records:
 * <u32 op><u32 imm><64 B vec>) on the real NNA and compare every NNDRD read-back with the
 * recorded one. Tells whether the silicon NN unit works under onna's backend (no soc-nna).
 *   hwreplay records.bin      (pins itself to CPU0) 
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mxu3.h"

#define R1(i) case i: NNRWR(i); break;
#define R8(b) R1(b) R1(b+1) R1(b+2) R1(b+3) R1(b+4) R1(b+5) R1(b+6) R1(b+7)
#define D1(i) case i: NNDRD(i); break;

static int rwr(unsigned imm)
{
	switch (imm) {
	R8(0x20) R8(0x28) R8(0x30) R8(0x38) R8(0x40) R8(0x48) R8(0x50) R8(0x58)
	default: return -1;
	}
	return 0;
}

static int dwr(unsigned imm)
{
	switch (imm) {
	case 0x00: NNDWR(0x00); break; case 0x02: NNDWR(0x02); break; case 0x04: NNDWR(0x04); break;
	case 0x07: NNDWR(0x07); break; case 0x20: NNDWR(0x20); break; case 0x21: NNDWR(0x21); break;
	case 0x40: NNDWR(0x40); break;
	default: return -1;
	}
	return 0;
}

static int drd(unsigned imm)
{
	switch (imm) {
	D1(0) D1(1) D1(2) D1(3) D1(4) D1(5) D1(6) D1(7) D1(8) D1(9) D1(10) D1(11) D1(12) D1(13)
	D1(14) D1(15) D1(0x20)
	default: return -1;
	}
	return 0;
}

static int mac(unsigned imm)
{
	switch (imm) {
	case 0: NNMAC(0); break; case 1: NNMAC(1); break; case 3: NNMAC(3); break;
	default: return -1;
	}
	return 0;
}

static int cmd(unsigned imm)
{
	switch (imm) {
	case 0x00: NNCMD(0x00); break; case 0x60: NNCMD(0x60); break; case 0x83: NNCMD(0x83); break;
	case 0x84: NNCMD(0x84); break; case 0x8b: NNCMD(0x8b); break; case 0xa0: NNCMD(0xa0); break;
	case 0xc0: NNCMD(0xc0); break; case 0xe0: NNCMD(0xe0); break;
	default: return -1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	FILE *f = fopen(argc > 1 ? argv[1] : "records.bin", "rb");
	struct { unsigned op, imm; unsigned char vec[64]; } r;
	unsigned char v[64] __attribute__((aligned(64)));
	long n = 0, reads = 0, bad = 0, first = -1;
	cpu_set_t cs;

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	if (!f || sched_setaffinity(0, sizeof(cs), &cs))
		return 2;
	while (fread(&r, sizeof(r), 1, f) == 1) {
		int e = 0;
		unsigned w;

		switch (r.op) {
		case 0: memcpy(&w, r.vec, 4); VWR0_SET(w); e = rwr(r.imm); break;
		case 2: memcpy(v, r.vec, 64); VPR0_LOAD(v); e = dwr(r.imm); break;
		case 3:
			e = drd(r.imm);
			VPR0_STORE(v);
			reads++;
			if (memcmp(v, r.vec, 64)) {
				if (first < 0)
					first = n;
				bad++;
			}
			break;
		case 4: memcpy(&w, r.vec, 4); VWR0_SET(w); e = mac(r.imm); break;
		case 5: e = cmd(r.imm); break;
		default: e = -1;
		}
		if (e) {
			fprintf(stderr, "record %ld: op %u imm 0x%x not supported\n", n, r.op, r.imm);
			return 2;
		}
		n++;
	}
	printf("%ld records, %ld read-backs, %ld differ (first at record %ld)\n", n, reads, bad, first);
	return bad != 0;
}
