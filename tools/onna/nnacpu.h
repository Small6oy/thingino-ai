/*
 * nnacpu: the CPU half of a venus-compatible T41 NNA runtime.
 *
 * Bit-exact with venus 0.1.2.5 (see docs/nna-runtime.md). Feature maps are HWC uint8 codes,
 * zero point 128. Build with -ffp-contract=off: float32 rounding order matters.
 */
#ifndef NNACPU_H
#define NNACPU_H
#include <stdint.h>

/* the per-layer SiLU table: 32 segments of 1024 t-units, de-interleaved */
struct nna_silu {
	int16_t dy[32];
	int16_t y[32];
};

/* venus stores it interleaved: (dy_i, dy_16+i) x16, then (y_i, y_16+i) x16 */
void nna_silu_load(struct nna_silu *t, const int16_t raw[64]);

/* q = clip(rint(PWL(rint(f32(S + B) * Fl)) / 128)), per pixel, per channel */
void nna_requant_silu(const int32_t *S, int npx, int noc, const int32_t *B, const float *Fl,
		      const struct nna_silu *t, uint8_t *q);

/* head: out = f32(S + B) * Fl */
void nna_head(const int32_t *S, int npx, int noc, const int32_t *B, const float *Fl, float *out);

/* concat input rescale: out = clip(rint((x - 128) * a) + 128) */
void nna_rescale(const uint8_t *x, int n, float a, uint8_t *out);

/* residual add as the NNA's QSUM computes it (CPU reference):
 * out = clip(floor((64 * (wa*a + wb*b) + B) / T) + 128) */
void nna_add_ref(const uint8_t *a, const uint8_t *b, int n, int wa, int wb, int32_t B, int32_t T,
		 uint8_t *out);

/* 5x5 max pool, stride 1, pad 2 (padding never wins) */
void nna_maxpool5(const uint8_t *x, int h, int w, int c, uint8_t *out);

/* nearest-neighbour x2 upsample */
void nna_upsample2(const uint8_t *x, int h, int w, int c, uint8_t *out);

#endif
