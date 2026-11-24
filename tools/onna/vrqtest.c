/* vrqtest: the MXU3 SiLU requant (nna_vrq_px) against the scalar LUT, for every SiLU table of a
 * venus .bin: all t in -40000..40000 (B = 0, Fl = 1), then random full-path cases with the layer's
 * own B and Fl. Runs on the camera or under user-mode QEMU -cpu XBurst2. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnabin.h"
#include "nnacpu.h"

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

int main(int argc, char **argv)
{
	struct nna_bin b;
	static struct nna_vrq v;
	static int32_t S[2][16] __attribute__((aligned(64)));
	static uint8_t q[64] __attribute__((aligned(64)));
	long cases = 0, bad = 0;
	int nrand = argc > 2 ? atoi(argv[2]) : 200000;

	if (argc < 2 || nna_bin_load(&b, argv[1]))
		return fprintf(stderr, "usage: vrqtest model.bin [random cases per table]\n"), 2;
	for (int k = 0; k < b.nconv; k++) {
		const struct nna_conv_rec *r = &b.conv[k];
		long lb = 0;

		if (!r->has_silu)
			continue;
		v.tb = r->silu;
		/* every t: S = t, B = 0, Fl = 1 (exact) */
		for (int c = 0; c < 16; c++)
			v.B[0][c] = v.B[1][c] = 0, v.Fl[0][c] = v.Fl[1][c] = 1.0f;
		nna_vrq_load(&v);
		for (long t0 = -40000; t0 <= 40000; t0 += 32) {
			for (int c = 0; c < 16; c++)
				S[0][c] = (int32_t)(t0 + c), S[1][c] = (int32_t)(t0 + 16 + c);
			nna_vrq_px(S[0], S[1], q);
			for (int c = 0; c < 32; c++) {
				long t = t0 + c, i = t < 0 ? 0 : t > 32767 ? 32768 : t + 1;

				cases++;
				if (q[c] != r->lut[i] && lb++ < 3)
					printf("conv %d t %ld: vec %u lut %u\n", k, t, q[c], r->lut[i]);
			}
		}
		/* the full path with this layer's B and Fl (sums as the NNA gives them) */
		for (int n = 0; n < nrand; n++) {
			int ob = (int)(rnd() % (unsigned)r->nob);
			uint8_t ref[32];

			for (int h = 0; h < 2; h++)
				for (int c = 0; c < 16; c++) {
					v.B[h][c] = r->B[32 * ob + 16 * h + c];
					v.Fl[h][c] = r->Fl[32 * ob + 16 * h + c];
					S[h][c] = (int32_t)(rnd() % 4000001u) - 2000000;
					if (rnd() % 4 == 0)	/* near the useful range */
						S[h][c] = (int32_t)(rnd() % 200001u) - 100000;
				}
			nna_vrq_load(&v);
			nna_vrq_px(S[0], S[1], q);
			nna_requant_silu(S[0], 1, 16, v.B[0], v.Fl[0], &r->silu, ref);
			nna_requant_silu(S[1], 1, 16, v.B[1], v.Fl[1], &r->silu, ref + 16);
			for (int c = 0; c < 32; c++) {
				cases++;
				if (q[c] != ref[c] && lb++ < 6)
					printf("conv %d full S %d: vec %u ref %u\n", k, S[c / 16][c % 16], q[c], ref[c]);
			}
		}
		bad += lb;
	}
	printf("vrqtest: %ld cases, %ld mismatches: %s\n", cases, bad, bad ? "FAIL" : "PASS");
	return bad != 0;
}
