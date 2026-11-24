/* nnaeng: the T41 NNA conv engine driver. See nnaeng.h.
 *
 * Tiling (v1, simple and correct; speed comes later): one MAC per 2x4 output tile (0x38 =
 * 0x12, as venus; the stem uses venus's one-row mode 0x3). For each tile the input window
 * ((sy + kh) rows x pitch pixels x planes) is written linearly into F RAM from address 0; the
 * engine's bounds check (0x2b/0x29 vs 0x2e/0x2c) supplies the padding.
 *
 * F RAM on silicon is sliced (as QEMU models only for the T40): a written byte is stored as
 * four 2-bit slices at lines l, l+d, l+2d, l+3d with d = 0x37/4, and a read sums them. So a
 * plane's data must fit in its first 0x37/4 lines, planes sit 0x37 lines apart, and each plane
 * is written separately from line k*0x37. (QEMU's T41 model stores bytes: the same result as
 * long as nothing collides.)
 * Weights: per 32-output-channel block, the groups (plane k, tap in snake order) go to W RAM
 * from entry 0. Sums: CPOC|CLROC, then 16 NNDRD = 8 pixels x 32 channels.
 */
#include <stdio.h>
#include <string.h>
#include "nnaeng.h"

#define ERR(...) do { fprintf(stderr, "nnaeng: " __VA_ARGS__); fputc('\n', stderr); return -1; } while (0)

enum {
	R_FY = 0x29, R_FX = 0x2b, R_FH = 0x2c, R_FW = 0x2e, R_PAD = 0x2f, R_WRD = 0x30,
	R_PLANE = 0x37, R_MACTL = 0x38, R_PITCH = 0x39, R_STRIDE = 0x3a, R_FMT = 0x3b,
	R_NPLANE = 0x3c, R_TAPS = 0x3e, R_PXMASK = 0x3f, R_MSF = 0x46, R_FLINE = 0x50,
	R_VEC = 0x55, R_WFCWN = 0x5a, R_WWR = 0x5c,
};
enum { C_CPOC = 1, C_CLROC = 2 };

/* the tap order of W groups: snake, row 1 reversed */
static const uint8_t snake[9][2] = {
	{ 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 2 }, { 1, 1 }, { 1, 0 }, { 2, 0 }, { 2, 1 }, { 2, 2 },
};

/* one W group (32 oc x 32 ic, 8-bit, msf 2) as 4 entries of 256 bytes: slice s = bits 2s..2s+1 */
static void put_group(const struct nna_be *be, const int8_t w[32][32])
{
	uint8_t ent[256];

	for (int s = 0; s < 4; s++) {
		for (int o = 0; o < 32; o++)
			for (int j = 0; j < 8; j++) {
				unsigned b = 0;

				for (int q = 0; q < 4; q++)
					b |= (((unsigned)(w[o][4 * j + q] + 128) >> (2 * s)) & 3) << (2 * q);
				ent[o * 8 + j] = (uint8_t)b;
			}
		be->dwr(be->ctx, 0x00, ent);
		be->dwr(be->ctx, 0x02, ent + 64);
		be->dwr(be->ctx, 0x04, ent + 128);
		be->dwr(be->ctx, 0x07, ent + 192);
	}
}

static void common_regs(const struct nna_be *be, int padval, int fw, int fh, int r21)
{
	be->cmd(be->ctx, 0);				/* reset */
	be->rwr(be->ctx, R_FMT, 0x33);			/* 8-bit W, 8-bit F */
	be->rwr(be->ctx, R_MSF, 2);			/* w = v - 128 */
	be->rwr(be->ctx, R_VEC, 2);			/* 32-byte pixels */
	be->rwr(be->ctx, R_MACTL, 0x03);		/* one row of 8 pixels (venus's stem value) */
	be->rwr(be->ctx, R_PXMASK, 0xff);
	be->rwr(be->ctx, R_PAD, (uint32_t)padval);
	be->rwr(be->ctx, R_FW, (uint32_t)fw);
	be->rwr(be->ctx, R_FH, (uint32_t)fh);
	/* registers venus writes that QEMU does not interpret, with venus's values. 0x21 and 0x23
	 * only for the stem (venus: 3 and 3); other layers keep their reset values */
	if (r21) {
		be->rwr(be->ctx, 0x21, (uint32_t)r21);
		be->rwr(be->ctx, 0x23, 3);
	}
	be->rwr(be->ctx, 0x28, 2);
}

/* after the weights, as venus does: the F write program (one word: count 1, step +1 line per
 * 4-pixel slot; QEMU does this unprogrammed, silicon needs it) and NNCMD 0x60 */
static void arm(const struct nna_be *be)
{
	be->rwr(be->ctx, 0x5b, 0);
	be->rwr(be->ctx, 0x45, 2);
	be->rwr(be->ctx, 0x58, 0);
	be->rwr(be->ctx, 0x59, 0x801);
	be->rwr(be->ctx, R_WFCWN, 1);
	be->cmd(be->ctx, 0x60);
}

/* CPOC|CLROC, then 16 NNDRD = 8 pixels x 32 channels. rows2: pixel p is (p/4, p%4) of a
 * 2x4 tile, else (0, p) of an 8-pixel row */
static void read_sums(const struct nna_be *be, int32_t *S, int OH, int OW, int noc, int oy, int ox0,
		      int ob, int rows2)
{
	uint8_t v[64];

	be->cmd(be->ctx, 0x80 | C_CPOC | C_CLROC);	/* 0x83, as venus */
	for (int k = 0; k < 16; k++) {
		int p = k / 2, y = oy + (rows2 ? p >> 2 : 0), ox = ox0 + (rows2 ? p & 3 : p);

		be->drd(be->ctx, (unsigned)k, v);
		if (ox >= OW || y >= OH)
			continue;
		for (int i = 0; i < 16; i++) {
			int oc = 32 * ob + 16 * (k % 2) + i;
			int32_t s;

			if (oc >= noc)
				continue;
			memcpy(&s, v + 4 * i, 4);
			S[((long)y * OW + ox) * noc + oc] = s;
		}
	}
}

/* write a run of 32-byte pixels into F RAM from line `line`, 2 pixels per NNDWR */
static void put_f_at(const struct nna_be *be, const uint8_t *px, int npx, int line)
{
	be->rwr(be->ctx, R_FLINE, (uint32_t)line);
	for (int i = 0; i < npx; i += 2)
		be->dwr(be->ctx, 0x20, px + 32L * i);
}

static void put_f(const struct nna_be *be, const uint8_t *px, int npx)
{
	put_f_at(be, px, npx, 0);
}

static int stem(const struct nna_be *be, const uint8_t *in, int H, int W, const struct nna_node *nd,
		int32_t *S, int OH, int OW)
{
	/* pixels = vertical strips of 6 RGBA rows (ic = 4*ky + c); taps along x: two MACs of 3,
	 * the second at word 3 with the W pointer advanced by the first (MAC imm bit 0) */
	enum { PITCH = 32 };
	static uint8_t f[PITCH * 32];
	int8_t w[32][32];

	common_regs(be, 0, W - 1, 0, 3);
	be->rwr(be->ctx, R_TAPS, 0x7);			/* (0,0) (0,1) (0,2) */
	be->rwr(be->ctx, R_NPLANE, 0);
	be->rwr(be->ctx, R_STRIDE, 0x11);		/* sx 2 (sy 2 as venus; one row) */
	be->rwr(be->ctx, R_PITCH, PITCH);
	be->rwr(be->ctx, R_PLANE, PITCH);
	be->rwr(be->ctx, R_WWR, 0);
	for (int kx = 0; kx < 6; kx++) {		/* groups kx 0..5 = entries 0..23 */
		memset(w, 0, sizeof(w));
		for (int o = 0; o < nd->oc; o++)
			for (int ky = 0; ky < 6; ky++)
				for (int c = 0; c < 4; c++)
					w[o][4 * ky + c] = nd->wq[(((long)ky * 6 + kx) * nd->oc + o) * nd->ic + c];
		put_group(be, w);
	}
	arm(be);
	for (int oy = 0; oy < OH; oy++)
		for (int ox0 = 0; ox0 < OW; ox0 += 8) {
			int x0 = 2 * ox0 - 2;

			memset(f, 0, sizeof(f));
			for (int cx = 0; cx < 20; cx++) {
				int x = x0 + cx;

				if (x < 0 || x >= W)
					continue;
				for (int ky = 0; ky < 6; ky++) {
					int y = 2 * oy - 2 + ky;

					if (y >= 0 && y < H)
						memcpy(&f[cx * 32 + 4 * ky], &in[((long)y * W + x) * 4], 4);
				}
			}
			put_f(be, f, PITCH);
			be->rwr(be->ctx, R_FY, 0);
			be->rwr(be->ctx, R_FX, (uint32_t)x0 & 0xfff);
			be->rwr(be->ctx, 0x31, 0);
			be->rwr(be->ctx, R_WRD, 0);
			be->mac(be->ctx, 1, 0);			/* kx 0..2, W groups 0..2 */
			be->rwr(be->ctx, R_FX, (uint32_t)(x0 + 3) & 0xfff);
			be->mac(be->ctx, 1, 3);			/* kx 3..5, W groups 3..5 */
			read_sums(be, S, OH, OW, nd->oc, oy, ox0, 0, 0);
		}
	return 0;
}

int nna_conv(const struct nna_be *be, const uint8_t *in, int H, int W, int C,
	     const struct nna_node *nd, int32_t *S, int *oh, int *ow)
{
	int st = nd->stride, pd = nd->pad, kh = nd->kh, kw = nd->kw;
	int OH = (H + 2 * pd - kh) / st + 1, OW = (W + 2 * pd - kw) / st + 1;
	int P = (C + 31) / 32, ntaps = kh * kw, pitch, plsz;
	unsigned taps = 0;
	static uint8_t f[4096 * 32];
	int8_t w[32][32];
	int order[9], nt = 0;

	*oh = OH, *ow = OW;
	if (nd->stem)
		return stem(be, in, H, W, nd, S, OH, OW);
	if (kh > 3 || kw > 3 || P > 16)
		ERR("conv %dx%d with %d planes does not fit", kh, kw, P);
	int rows = st + kh;				/* input rows of a 2-row output tile */

	pitch = ((3 * st + kw) + 3) & ~3;
	plsz = rows * pitch;			/* data pixels per plane = 0x37 (lines per plane) */
	if (pitch > 60 || plsz > 508 || P * plsz > 1024 || P * ntaps * 4 > 1024)
		ERR("tile does not fit (pitch %d, plane %d, planes %d)", pitch, plsz, P);
	for (int g = 0; g < 9; g++)
		if (snake[g][0] < kh && snake[g][1] < kw) {
			taps |= 1u << (snake[g][0] * 3 + snake[g][1]);
			order[nt++] = g;
		}
	common_regs(be, nd->padval, W - 1, H - 1, 0);
	be->rwr(be->ctx, R_MACTL, 0x12);		/* 2 rows x 4 pixels */
	be->rwr(be->ctx, R_TAPS, taps);
	be->rwr(be->ctx, R_NPLANE, (uint32_t)(P - 1));
	be->rwr(be->ctx, R_STRIDE, st == 2 ? 0x11 : 0);
	be->rwr(be->ctx, R_PITCH, (uint32_t)pitch);
	be->rwr(be->ctx, R_PLANE, (uint32_t)plsz);
	for (int ob = 0; ob < (nd->oc + 31) / 32; ob++) {
		be->rwr(be->ctx, R_WWR, 0);
		for (int k = 0; k < P; k++)
			for (int i = 0; i < nt; i++) {
				int ky = snake[order[i]][0], kx = snake[order[i]][1];

				memset(w, 0, sizeof(w));
				for (int o = 0; o < 32 && 32 * ob + o < nd->oc; o++)
					for (int c = 0; c < 32 && 32 * k + c < C; c++)
						w[o][c] = nd->wq[(((long)ky * kw + kx) * nd->oc + 32 * ob + o) * nd->ic + 32 * k + c];
				put_group(be, w);
			}
		arm(be);
		for (int oy = 0; oy < OH; oy += 2)
			for (int ox0 = 0; ox0 < OW; ox0 += 4) {
				int y0 = oy * st - pd, x0 = ox0 * st - pd;

				memset(f, nd->padval, (size_t)P * plsz * 32);
				for (int k = 0; k < P; k++)
					for (int r = 0; r < rows; r++) {
						int y = y0 + r;

						if (y < 0 || y >= H)
							continue;
						for (int cx = 0; cx < pitch; cx++) {
							int x = x0 + cx, n = C - 32 * k < 32 ? C - 32 * k : 32;
							uint8_t *d = &f[((long)k * plsz + r * pitch + cx) * 32];

							if (x < 0 || x >= W)
								continue;
							memcpy(d, &in[((long)y * W + x) * C + 32 * k], (size_t)n);
							memset(d + n, 0, (size_t)(32 - n));
						}
					}
				for (int k = 0; k < P; k++)	/* plane k from line k*0x37 */
					put_f_at(be, f + (long)k * plsz * 32, plsz, k * plsz);
				be->rwr(be->ctx, R_FY, (uint32_t)y0 & 0xfff);
				be->rwr(be->ctx, R_FX, (uint32_t)x0 & 0xfff);
				be->rwr(be->ctx, 0x31, 0);
				be->rwr(be->ctx, R_WRD, 0);
				be->mac(be->ctx, 1, 0);
				read_sums(be, S, OH, OW, nd->oc, oy, ox0, ob, 1);
			}
	}
	return 0;
}
