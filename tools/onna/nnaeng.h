/*
 * nnaeng: drive the T41 NNA's convolution engine for one conv layer.
 * Semantics: docs/nna-runtime.md (engine registers, W entries, F RAM, MAC, CPOC, NNDRD).
 * The engine is reached through a backend: the real MXU3 NN instructions on the camera
 * (be_mxu3.c), or a test oracle on the host.
 */
#ifndef NNAENG_H
#define NNAENG_H
#include <stdint.h>
#include "nnabin.h"

struct nna_be {
	void *ctx;
	void (*rwr)(void *ctx, unsigned reg, uint32_t v);		/* NNRWR */
	void (*dwr)(void *ctx, unsigned imm, const uint8_t v[64]);	/* NNDWR */
	void (*drd)(void *ctx, unsigned imm, uint8_t v[64]);		/* NNDRD */
	void (*mac)(void *ctx, unsigned imm, uint32_t word);		/* NNMAC */
	void (*cmd)(void *ctx, unsigned imm);				/* NNCMD */
};

/* requant fused into the read-back: each NNDRD's 16 sums become output bytes at once */
struct nna_rq {
	const int32_t *B;
	const float *Fl;
	const uint8_t *lut;	/* NNA_SILU_LUT bytes */
	uint8_t *q;		/* OH x OW x oc output codes */
};

/* one conv on the NNA: in = HWC uint8 codes (C channels). With rq == NULL the int32 sums go to
 * S (OH x OW x oc); with rq (oc a multiple of 16) they are requantized straight into rq->q.
 * 0 on success, -1 if the layer does not fit this tiling (message on stderr). */
int nna_conv(const struct nna_be *be, const uint8_t *in, int H, int W, int C,
	     const struct nna_node *nd, int32_t *S, const struct nna_rq *rq, int *oh, int *ow);

/* instruction counts per type (rwr, dwr, drd, mac, cmd) through a wrapped backend */
void nna_count_wrap(struct nna_be *be);
void nna_count_get(long c[5]);

#endif
