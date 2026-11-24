/* nnacpu: the CPU half of a venus-compatible T41 NNA runtime. See nnacpu.h. */
#include <math.h>
#include <string.h>
#include "nnacpu.h"

/* rint (ties to even) without a libm call: for |x| < 2^22, adding and subtracting 1.5 * 2^23 in
 * float32 rounds to the nearest integer, ties to even (the FPU's default mode), exactly as rint */
static inline long frint(float x)
{
	if (x > -4194304.0f && x < 4194304.0f) {
		const float m = 12582912.0f;		/* not folded without -ffast-math */

		return (long)((x + m) - m);
	}
	return lrintf(x);
}

static inline uint8_t clip8(long v)
{
	return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

void nna_silu_load(struct nna_silu *t, const int16_t raw[64])
{
	for (int i = 0; i < 16; i++) {
		t->dy[i] = raw[2 * i];
		t->dy[16 + i] = raw[2 * i + 1];
		t->y[i] = raw[32 + 2 * i];
		t->y[16 + i] = raw[32 + 2 * i + 1];
	}
}

void nna_requant_silu(const int32_t *S, int npx, int noc, const int32_t *B, const float *Fl,
		      const struct nna_silu *tb, uint8_t *q)
{
	for (int p = 0; p < npx; p++) {
		for (int c = 0; c < noc; c++) {
			float v = (float)(S[(long)p * noc + c] + B[c]) * Fl[c];
			long t = frint(v);
			long seg = t >> 10;			/* floor, also for t < 0 */
			float y;

			if (seg < 0) {
				y = (float)tb->y[0];
			} else {
				long sc = seg > 31 ? 31 : seg;
				long tc = t > 32767 ? 32767 : t;
				float fr = (float)(tc - sc * 1024);
				float inc = (float)tb->dy[sc] * fr;

				inc = inc / 1024.0f;
				y = (float)tb->y[sc] + inc;
			}
			q[(long)p * noc + c] = clip8(frint(y / 128.0f));
		}
	}
}

void nna_head(const int32_t *S, int npx, int noc, const int32_t *B, const float *Fl, float *out)
{
	for (long i = 0; i < (long)npx * noc; i++)
		out[i] = (float)(S[i] + B[i % noc]) * Fl[i % noc];
}

void nna_rescale(const uint8_t *x, int n, float a, uint8_t *out)
{
	for (int i = 0; i < n; i++)
		out[i] = clip8(frint(((float)x[i] - 128.0f) * a) + 128);
}

void nna_add_ref(const uint8_t *a, const uint8_t *b, int n, int wa, int wb, int32_t B, int32_t T,
		 uint8_t *out)
{
	for (int i = 0; i < n; i++) {
		long long num = 64LL * (wa * a[i] + wb * b[i]) + B;
		long long qq = num / T;

		if ((num % T) && ((num < 0) != (T < 0)))
			qq--;				/* floor division */
		out[i] = clip8((long)qq + 128);
	}
}

void nna_maxpool5(const uint8_t *x, int h, int w, int c, uint8_t *out)
{
	for (int y = 0; y < h; y++)
		for (int xx = 0; xx < w; xx++)
			for (int k = 0; k < c; k++) {
				int m = 0;

				for (int dy = -2; dy <= 2; dy++) {
					int yy = y + dy;

					if (yy < 0 || yy >= h)
						continue;
					for (int dx = -2; dx <= 2; dx++) {
						int x2 = xx + dx;
						int v;

						if (x2 < 0 || x2 >= w)
							continue;
						v = x[((long)yy * w + x2) * c + k];
						if (v > m)
							m = v;
					}
				}
				out[((long)y * w + xx) * c + k] = (uint8_t)m;
			}
}

void nna_upsample2(const uint8_t *x, int h, int w, int c, uint8_t *out)
{
	for (int y = 0; y < 2 * h; y++)
		for (int xx = 0; xx < 2 * w; xx++)
			memcpy(&out[((long)y * 2 * w + xx) * c], &x[((long)(y / 2) * w + xx / 2) * c], c);
}

void nna_silu_lut(const struct nna_silu *tb, uint8_t lut[NNA_SILU_LUT])
{
	/* the same float32 code as nna_requant_silu, for every t in -1 (= any t < 0) .. 32767 */
	for (long t = -1; t <= 32767; t++) {
		float y;

		if (t < 0) {
			y = (float)tb->y[0];
		} else {
			long sc = (t >> 10) > 31 ? 31 : t >> 10;
			float inc = (float)tb->dy[sc] * (float)(t - sc * 1024);

			inc = inc / 1024.0f;
			y = (float)tb->y[sc] + inc;
		}
		lut[t + 1] = clip8(frint(y / 128.0f));
	}
}

void nna_requant_lut(const int32_t *S, int npx, int noc, const int32_t *B, const float *Fl,
		     const uint8_t lut[NNA_SILU_LUT], uint8_t *q)
{
	for (int p = 0; p < npx; p++) {
		const int32_t *s = S + (long)p * noc;
		uint8_t *o = q + (long)p * noc;

		for (int c = 0; c < noc; c++) {
			long t = frint((float)(s[c] + B[c]) * Fl[c]);

			t = t < 0 ? -1 : t > 32767 ? 32767 : t;
			o[c] = lut[t + 1];
		}
	}
}

void nna_add_lut(int wa, int wb, int32_t B, int32_t T, uint8_t lut[65536])
{
	static uint8_t av[65536], bv[65536];

	for (int i = 0; i < 65536; i++)
		av[i] = (uint8_t)(i >> 8), bv[i] = (uint8_t)i;
	nna_add_ref(av, bv, 65536, wa, wb, B, T, lut);
}

void nna_add_apply(const uint8_t *a, const uint8_t *b, int n, const uint8_t lut[65536], uint8_t *out)
{
	for (int i = 0; i < n; i++)
		out[i] = lut[(unsigned)a[i] << 8 | b[i]];
}
