/* ojzdlrun: run a JZDL v5 model with ojzdl, same CLI and output files as
 * ../jzdl/jzdlrun.cpp (libjzdl), so the two can be compared byte for byte.
 *
 *   ojzdlrun [-n RUNS] [-t] MODEL.bin INPUT.rgb OUTPREFIX [blob ...]
 *
 * INPUT.rgb: in_h x in_w x 3 uint8, HWC. Default blob: the model output,
 * written as OUTPREFIX.-999.f32 (int32 w, h, c, then floats), like jzdlrun.
 * Other blobs are written as floats too (int8 values converted), same header.
 * -n RUNS: run forward RUNS times, print min/avg ms. -t: per-layer times (min).
 */
#include "ojzdl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OUT_BLOB_ALIAS (-999) /* jzdlrun's "model output" blob id */

static uint64_t now_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static int dump(const char *prefix, int id, int w, int h, int c, const float *f,
                const int8_t *i8, size_t n)
{
	char name[4096];
	int32_t hdr[3] = { w, h, c };
	FILE *o;
	size_t i;
	int rc = 0;

	snprintf(name, sizeof(name), "%s.%d.f32", prefix, id);
	o = fopen(name, "wb");
	if (!o) {
		fprintf(stderr, "cannot write %s\n", name);
		return -1;
	}
	if (fwrite(hdr, 4, 3, o) != 3)
		rc = -1;
	if (f) {
		if (fwrite(f, sizeof(float), n, o) != n)
			rc = -1;
	} else {
		for (i = 0; i < n && !rc; i++) {
			float v = (float)i8[i];
			if (fwrite(&v, sizeof(v), 1, o) != 1)
				rc = -1;
		}
	}
	if (fclose(o))
		rc = -1;
	printf("blob %d: w %d h %d c %d total %zu -> %s%s\n", id, w, h, c, n, name, rc ? " (WRITE ERROR)" : "");
	return rc;
}

int main(int argc, char **argv)
{
	int runs = 1, timing = 0, a = 1, i, r, rc = 1;
	const char *err = NULL;
	ojzdl_net *n;
	uint8_t *rgb = NULL;
	size_t insz;
	uint64_t best = UINT64_MAX, total = 0;
	uint64_t *lbest = NULL;
	FILE *f;

	while (a < argc && argv[a][0] == '-' && argv[a][1] && !(argv[a][1] >= '0' && argv[a][1] <= '9')) {
		if (!strcmp(argv[a], "-n") && a + 1 < argc)
			runs = atoi(argv[a += 1]);
		else if (!strcmp(argv[a], "-t"))
			timing = 1;
		else
			break;
		a++;
	}
	if (argc - a < 3 || runs < 1) {
		fprintf(stderr, "usage: %s [-n RUNS] [-t] MODEL.bin INPUT.rgb OUTPREFIX [blob ...]\n", argv[0]);
		return 2;
	}
	n = ojzdl_load_file(argv[a], &err);
	if (!n) {
		fprintf(stderr, "cannot load %s: %s\n", argv[a], err ? err : "?");
		return 1;
	}
	insz = (size_t)ojzdl_in_h(n) * ojzdl_in_w(n) * ojzdl_in_c(n);
	printf("input %dx%dx%d (w h c), %d layers\n", ojzdl_in_w(n), ojzdl_in_h(n), ojzdl_in_c(n),
	       ojzdl_layer_count(n));
	{
		size_t ar, sc;
		ojzdl_memory(n, &ar, &sc);
		printf("memory: arena %zu B, scratch %zu B\n", ar, sc);
	}
	rgb = malloc(insz);
	f = fopen(argv[a + 1], "rb");
	if (!rgb || !f || fread(rgb, 1, insz, f) != insz) {
		fprintf(stderr, "cannot read %zu bytes from %s\n", insz, argv[a + 1]);
		if (f)
			fclose(f);
		goto out;
	}
	fclose(f);

	if (argc - a > 3 && ojzdl_keep_blobs(n, 1)) {
		fprintf(stderr, "out of memory\n");
		goto out;
	}
	if (getenv("OJZDL_SIMD") && !strcmp(getenv("OJZDL_SIMD"), "0"))
		ojzdl_set_simd(n, 0);
	ojzdl_set_timing(n, timing);
	lbest = calloc((size_t)ojzdl_layer_count(n), sizeof(*lbest));
	if (!lbest)
		goto out;
	for (i = 0; i < ojzdl_layer_count(n); i++)
		lbest[i] = UINT64_MAX;

	for (r = 0; r < runs; r++) {
		uint64_t t0 = now_ns(), dt;
		if (ojzdl_input_rgb(n, rgb) || ojzdl_forward(n)) {
			fprintf(stderr, "forward failed\n");
			goto out;
		}
		dt = now_ns() - t0;
		total += dt;
		if (dt < best)
			best = dt;
		for (i = 0; timing && i < ojzdl_layer_count(n); i++) {
			int t;
			uint64_t ns;
			ojzdl_layer_info(n, i, &t, &ns);
			if (ns < lbest[i])
				lbest[i] = ns;
		}
	}
	printf("forward: %d runs, min %.3f ms, avg %.3f ms\n", runs, best / 1e6, total / 1e6 / runs);
	for (i = 0; timing && i < ojzdl_layer_count(n); i++) {
		int t;
		uint64_t ns;
		ojzdl_layer_info(n, i, &t, &ns);
		printf("  layer %3d t%-3d %9.3f ms\n", i, t, lbest[i] / 1e6);
	}

	if (argc - a == 3) {
		size_t cnt;
		const float *o = ojzdl_output(n, &cnt);
		if (dump(argv[a + 2], OUT_BLOB_ALIAS, 1, 1, (int)cnt, o, NULL, cnt))
			goto out;
	}
	for (i = a + 3; i < argc; i++) {
		int id = atoi(argv[i]), h, w, c, bits;
		const int8_t *i8;
		const float *f32;
		if (id == OUT_BLOB_ALIAS) {
			size_t cnt;
			const float *o = ojzdl_output(n, &cnt);
			if (dump(argv[a + 2], id, 1, 1, (int)cnt, o, NULL, cnt))
				goto out;
			continue;
		}
		if (ojzdl_blob(n, id, &h, &w, &c, &bits, &i8, &f32)) {
			fprintf(stderr, "no blob %d\n", id);
			goto out;
		}
		if (dump(argv[a + 2], id, w, h, c, f32, i8, (size_t)h * w * c))
			goto out;
	}
	rc = 0;
out:
	free(lbest);
	free(rgb);
	ojzdl_free(n);
	return rc;
}
