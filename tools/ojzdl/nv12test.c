/* nv12test: ojzdl_resize_nv12 against the per-pixel NV12 -> RGB of raptor-hal's
 * backends (hal_ivs_jzdl.cpp nv12_to_rgb) fed row by row to ojzdl_resize_rows.
 * The output must be identical byte for byte; prints the times of both.
 *
 *   nv12test [-v] [W H DW DH [RUNS]]   default 640 360 416 320 100
 *
 * Checks every size in the table below on 40 frames each (black/white
 * patterns, saturated chroma, noise), then times W x H -> DW x DH.
 * Exit status 1 when any output differs. -v prints a hash of every case's output,
 * to compare builds (scalar, MXU2, MXU3) with each other.
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#ifdef OJZDL_MXU3
#define _GNU_SOURCE /* sched_setaffinity: the MXU3 is used on core 0 only, as onna does */
#include <sched.h>
#endif
#include "ojzdl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t clamp_u8(int v)
{
	return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v);
}

/* the per-pixel form, as in raptor-hal */
static void nv12_row_to_rgb(const uint8_t *nv12, uint8_t *rgb, int w, int h, int row)
{
	const uint8_t *yp = nv12 + (size_t)row * w;
	const uint8_t *uvp = nv12 + (size_t)w * h + (size_t)(row / 2) * w;
	int col;
	for (col = 0; col < w; col++) {
		int y = yp[col], u = uvp[col & ~1] - 128, v = uvp[col | 1] - 128;
		rgb[col * 3 + 0] = clamp_u8(y + ((359 * v) >> 8));
		rgb[col * 3 + 1] = clamp_u8(y - ((88 * u + 183 * v) >> 8));
		rgb[col * 3 + 2] = clamp_u8(y + ((454 * u) >> 8));
	}
}

static int verbose;

struct ctx {
	const uint8_t *nv12;
	uint8_t *row;
	int w, h;
};

static const uint8_t *get_row(void *p, int y)
{
	struct ctx *c = p;
	nv12_row_to_rgb(c->nv12, c->row, c->w, c->h, y);
	return c->row;
}

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void fill(uint8_t *f, size_t n, int kind, uint32_t seed)
{
	size_t k;
	for (k = 0; k < n; k++) {
		seed = seed * 1103515245u + 12345u;
		f[k] = kind == 0 ? (k % 3 ? 0 : 255) : kind == 1 ? 255 : kind == 2 ? 0 : (uint8_t)(seed >> 24);
	}
}

/* 1 when both paths give the same bytes */
static int same(int w, int h, int dw, int dh, int kind, uint32_t seed)
{
	/* NV12: h luma rows, then (h + 1) / 2 chroma rows; + 2: an odd last column reads uv[col | 1]
	 * one past the row */
	size_t fsz = (size_t)w * h + (size_t)w * ((h + 1) / 2) + 2;
	size_t osz = (size_t)dw * dh * 3;
	uint8_t *f = malloc(fsz), *o1 = malloc(osz), *o2 = malloc(osz), *row = malloc((size_t)w * 3);
	ojzdl_resize *r = ojzdl_resize_create(w, h, dw, dh);
	struct ctx c;
	int ok = 0;
	if (f && o1 && o2 && row && r) {
		fill(f, fsz, kind, seed);
		c.nv12 = f;
		c.row = row;
		c.w = w;
		c.h = h;
		ok = !ojzdl_resize_rows(r, get_row, &c, o1) && !ojzdl_resize_nv12(r, f, o2) &&
		     !memcmp(o1, o2, osz);
		if (verbose) {
			uint32_t hsh = 2166136261u;
			size_t k;
			for (k = 0; k < osz; k++)
				hsh = (hsh ^ o2[k]) * 16777619u;
			printf("%dx%d -> %dx%d kind %d seed %u: %08x\n", w, h, dw, dh, kind, (unsigned)seed,
			       (unsigned)hsh);
		}
	}
	ojzdl_resize_free(r);
	free(f);
	free(o1);
	free(o2);
	free(row);
	return ok;
}

int main(int argc, char **argv)
{
	static const int sz[][4] = {
		{640, 360, 416, 320}, {640, 360, 448, 256}, {640, 360, 384, 224}, {640, 360, 320, 192},
		{640, 360, 320, 416}, {640, 360, 416, 234}, {640, 360, 640, 360}, {700, 400, 416, 320},
		{999, 361, 416, 320}, {1280, 720, 416, 320}, {1280, 720, 448, 256}, {213, 120, 416, 320},
		{641, 361, 416, 320}, {639, 359, 383, 223}, {2, 2, 1, 1}, {3, 5, 7, 9}, {1920, 1080, 416, 320},
	};
	int z, s, i, w, h, dw, dh, runs, cases = 0, bad = 0;
	double t0, a, t_rows = 1e9, t_nv12 = 1e9;
	size_t fsz;
	uint8_t *f, *out;
	ojzdl_resize *r;
	struct ctx c;

#ifdef OJZDL_MXU3
	{
		cpu_set_t cs;
		CPU_ZERO(&cs);
		CPU_SET(0, &cs);
		if (sched_setaffinity(0, sizeof(cs), &cs)) {
			fprintf(stderr, "cannot pin to core 0\n");
			return 2;
		}
	}
#endif
	if (argc > 1 && !strcmp(argv[1], "-v")) {
		verbose = 1;
		argc--;
		argv++;
	}
	w = argc > 4 ? atoi(argv[1]) : 640;
	h = argc > 4 ? atoi(argv[2]) : 360;
	dw = argc > 4 ? atoi(argv[3]) : 416;
	dh = argc > 4 ? atoi(argv[4]) : 320;
	runs = argc > 5 ? atoi(argv[5]) : 100;
	fsz = (size_t)w * h + (size_t)w * ((h + 1) / 2) + 2;

	for (z = 0; z < (int)(sizeof(sz) / sizeof(sz[0])); z++)
		for (s = 0; s < 40; s++) {
			cases++;
			if (!same(sz[z][0], sz[z][1], sz[z][2], sz[z][3], s < 3 ? s : 3, 7u + 977u * s + z)) {
				bad++;
				printf("DIFFERENT: %dx%d -> %dx%d, frame %d\n", sz[z][0], sz[z][1], sz[z][2], sz[z][3], s);
			}
		}
	printf("ojzdl_resize_nv12 == per-pixel NV12 + ojzdl_resize_rows: %d/%d cases\n", cases - bad, cases);

	f = malloc(fsz);
	out = malloc((size_t)dw * dh * 3);
	c.row = malloc((size_t)w * 3);
	r = ojzdl_resize_create(w, h, dw, dh);
	if (!f || !out || !c.row || !r)
		return 2;
	fill(f, fsz, 3, 1);
	c.nv12 = f;
	c.w = w;
	c.h = h;
	for (i = 0; i < runs; i++) {
		t0 = now_ms();
		ojzdl_resize_rows(r, get_row, &c, out);
		a = now_ms() - t0;
		t_rows = a < t_rows ? a : t_rows;
		t0 = now_ms();
		ojzdl_resize_nv12(r, f, out);
		a = now_ms() - t0;
		t_nv12 = a < t_nv12 ? a : t_nv12;
	}
	printf("%dx%d -> %dx%d, best of %d: per-pixel rows %.2f ms, ojzdl_resize_nv12 %.2f ms\n", w, h, dw,
	       dh, runs, t_rows, t_nv12);
	ojzdl_resize_free(r);
	free(f);
	free(out);
	free(c.row);
	return bad ? 1 : 0;
}
