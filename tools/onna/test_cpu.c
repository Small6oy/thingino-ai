/* test_cpu: run nnacpu.c over golden.py's per-op test vectors (GOLDEN_DUMP) and compare bytes.
 * usage: test_cpu <dump dir> */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnacpu.h"

static void *slurp(const char *dir, const char *name, long *len)
{
	char p[1024];
	FILE *f;
	void *b;

	snprintf(p, sizeof(p), "%s/%s", dir, name);
	f = fopen(p, "rb");
	if (!f) {
		perror(p);
		exit(2);
	}
	fseek(f, 0, SEEK_END);
	*len = ftell(f);
	rewind(f);
	b = malloc(*len ? *len : 1);
	if (fread(b, 1, *len, f) != (size_t)*len)
		exit(2);
	fclose(f);
	return b;
}

static int cmpz(const char *dir, const void *got, const void *want, long len)
{
	long bad = 0;

	for (long i = 0; i < len; i++)
		bad += ((const uint8_t *)got)[i] != ((const uint8_t *)want)[i];
	printf("%-28s %s (%ld/%ld bytes differ)\n", dir, bad ? "FAIL" : "ok", bad, len);
	return bad != 0;
}

static int cmpstr(const void *a, const void *b)
{
	return strcmp(*(char *const *)a, *(char *const *)b);
}

int main(int argc, char **argv)
{
	DIR *d = opendir(argc > 1 ? argv[1] : "gold");
	struct dirent *e;
	char *names[512];
	int n = 0, fails = 0;

	if (!d) {
		perror("opendir");
		return 2;
	}
	while ((e = readdir(d)) && n < 512)
		if (e->d_name[0] != '.')
			names[n++] = strdup(e->d_name);
	closedir(d);
	qsort(names, n, sizeof(*names), cmpstr);
	for (int i = 0; i < n; i++) {
		char dir[1024], kind[32];
		long l, ln[6];
		char *meta;
		long a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;
		double fa = 0;

		snprintf(dir, sizeof(dir), "%s/%s", argc > 1 ? argv[1] : "gold", names[i]);
		meta = slurp(dir, "meta.txt", &l);
		meta[l - 1] = 0;
		sscanf(meta, "%31s", kind);
		if (!strcmp(kind, "requant") || !strcmp(kind, "head")) {
			int32_t *S, *B;
			float *Fl;

			sscanf(meta, "%*s %ld %ld", &a0, &a1);
			S = slurp(dir, "S.bin", &ln[0]);
			B = slurp(dir, "B.bin", &ln[1]);
			Fl = slurp(dir, "Fl.bin", &ln[2]);
			if (!strcmp(kind, "requant")) {
				int16_t *tab = slurp(dir, "table.bin", &ln[3]), raw[64];
				uint8_t *want = slurp(dir, "q.bin", &ln[4]), *got = malloc(ln[4]);
				struct nna_silu t;

				for (int k = 0; k < 16; k++) {	/* back to venus's interleaved order */
					raw[2 * k] = tab[k];
					raw[2 * k + 1] = tab[16 + k];
					raw[32 + 2 * k] = tab[32 + k];
					raw[32 + 2 * k + 1] = tab[48 + k];
				}
				nna_silu_load(&t, raw);
				nna_requant_silu(S, (int)a0, (int)a1, B, Fl, &t, got);
				fails += cmpz(names[i], got, want, ln[4]);
				{	/* the table path must give the same bytes */
					static uint8_t lut[NNA_SILU_LUT];

					nna_silu_lut(&t, lut);
					memset(got, 0, (size_t)ln[4] & 0x7fffffff);
					nna_requant_lut(S, (int)a0, (int)a1, B, Fl, lut, got);
					fails += cmpz("  (lut)", got, want, ln[4]);
				}
			} else {
				float *want = slurp(dir, "out.bin", &ln[4]), *got = malloc(ln[4]);

				nna_head(S, (int)a0, (int)a1, B, Fl, got);
				fails += cmpz(names[i], got, want, ln[4]);
			}
		} else if (!strcmp(kind, "add")) {
			uint8_t *a, *b, *want, *got;

			sscanf(meta, "%*s %ld %ld %ld %ld %ld", &a0, &a1, &a2, &a3, &a4);
			a = slurp(dir, "a.bin", &ln[0]);
			b = slurp(dir, "b.bin", &ln[1]);
			want = slurp(dir, "out.bin", &ln[2]);
			got = malloc(ln[2]);
			nna_add_ref(a, b, (int)a0, (int)a1, (int)a2, (int32_t)a3, (int32_t)a4, got);
			fails += cmpz(names[i], got, want, ln[2]);
			{
				static uint8_t lut[65536];

				nna_add_lut((int)a1, (int)a2, (int32_t)a3, (int32_t)a4, lut);
				nna_add_apply(a, b, (int)a0, lut, got);
				fails += cmpz("  (lut)", got, want, ln[2]);
			}
		} else if (!strcmp(kind, "rescale")) {
			uint8_t *x, *want, *got;

			sscanf(meta, "%*s %ld %lf", &a0, &fa);
			x = slurp(dir, "x.bin", &ln[0]);
			want = slurp(dir, "out.bin", &ln[1]);
			got = malloc(ln[1]);
			nna_rescale(x, (int)a0, (float)fa, got);
			fails += cmpz(names[i], got, want, ln[1]);
		} else if (!strcmp(kind, "maxpool5") || !strcmp(kind, "upsample2")) {
			uint8_t *x, *want, *got;

			sscanf(meta, "%*s %ld %ld %ld", &a0, &a1, &a2);
			x = slurp(dir, "x.bin", &ln[0]);
			want = slurp(dir, "out.bin", &ln[1]);
			got = malloc(ln[1]);
			if (kind[0] == 'm')
				nna_maxpool5(x, (int)a0, (int)a1, (int)a2, got);
			else
				nna_upsample2(x, (int)a0, (int)a1, (int)a2, got);
			fails += cmpz(names[i], got, want, ln[1]);
		} else {
			printf("%-28s skipped (%s)\n", names[i], kind);
		}
	}
	printf("%d of %d ops differ\n", fails, n);
	return fails != 0;
}
