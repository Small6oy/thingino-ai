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

/* one conv on the NNA: in = HWC uint8 codes (C channels), S = OH x OW x oc int32 sums.
 * 0 on success, -1 if the layer does not fit this tiling (message on stderr). */
int nna_conv(const struct nna_be *be, const uint8_t *in, int H, int W, int C,
	     const struct nna_node *nd, int32_t *S, int *oh, int *ow);

#endif
