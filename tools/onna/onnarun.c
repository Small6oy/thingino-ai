/*
 * onnarun: run a venus .bin with onna. Writes <prefix>0.f32 .. for the heads.
 *   onnarun [-e cpu|oracle|nna] [-c] model.bin model.graph input_rgba.raw prefix
 *   -e  where convs run: cpu (plain integer loop), oracle (the NNA driver against the host
 *       test oracle; builds with ORACLE=1) or nna (the real NNA; T41 build, pins itself to
 *       CPU0). Default cpu.
 *   -c  also run the CPU conv and compare every layer's sums.
 */
#ifdef ONNA_MXU3
#define _GNU_SOURCE		/* sched_setaffinity: the NN unit is on CPU0 only */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nnabin.h"
#include "nnaeng.h"
#ifdef ONNA_ORACLE
#include "be_oracle.h"
#endif
#ifdef ONNA_MXU3
#include "be_mxu3.h"
#include <sched.h>
#endif

struct tensor {
	int h, w, c;
	uint8_t *u8;
	float *f;
};

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* integer conv on uint8 codes: S[oy][ox][oc] (int32) */
static int32_t *conv_cpu(const struct tensor *in, const struct nna_node *nd, int *oh, int *ow)
{
	int OH = (in->h + 2 * nd->pad - nd->kh) / nd->stride + 1;
	int OW = (in->w + 2 * nd->pad - nd->kw) / nd->stride + 1;
	int32_t *S = malloc(sizeof(int32_t) * OH * OW * nd->oc);
	uint8_t padv[512];

	memset(padv, nd->padval, sizeof(padv));
	for (int oy = 0; oy < OH; oy++)
		for (int ox = 0; ox < OW; ox++) {
			int32_t *acc = &S[((long)oy * OW + ox) * nd->oc];

			memset(acc, 0, sizeof(int32_t) * nd->oc);
			for (int ky = 0; ky < nd->kh; ky++)
				for (int kx = 0; kx < nd->kw; kx++) {
					int iy = oy * nd->stride + ky - nd->pad, ix = ox * nd->stride + kx - nd->pad;
					const uint8_t *px = (iy < 0 || iy >= in->h || ix < 0 || ix >= in->w) ? padv :
						&in->u8[((long)iy * in->w + ix) * in->c];
					const int8_t *wr = &nd->wq[((long)ky * nd->kw + kx) * nd->oc * nd->ic];

					for (int o = 0; o < nd->oc; o++) {
						int32_t s = 0;
						const int8_t *wo = wr + (long)o * nd->ic;

						for (int i = 0; i < nd->ic; i++)
							s += wo[i] * px[i];
						acc[o] += s;
					}
				}
		}
	*oh = OH, *ow = OW;
	return S;
}

int main(int argc, char **argv)
{
	struct nna_bin b;
	struct nna_graph g;
	struct tensor *T;
	FILE *f;
	double t0;

	struct nna_be be;
	int use_nna = 0, check = 0, ai = 1, bad_layers = 0;

	while (ai < argc && argv[ai][0] == '-') {
		if (!strcmp(argv[ai], "-c"))
			check = 1, ai++;
		else if (!strcmp(argv[ai], "-e") && ai + 1 < argc) {
			if (!strcmp(argv[ai + 1], "oracle")) {
#ifdef ONNA_ORACLE
				nna_be_oracle(&be);
				use_nna = 1;
#else
				fprintf(stderr, "built without the oracle (make ORACLE=1)\n");
				return 2;
#endif
			} else if (!strcmp(argv[ai + 1], "nna")) {
#ifdef ONNA_MXU3
				cpu_set_t cs;

				CPU_ZERO(&cs);
				CPU_SET(0, &cs);
				if (sched_setaffinity(0, sizeof(cs), &cs)) {
					perror("sched_setaffinity CPU0");
					return 1;
				}
				nna_be_mxu3(&be);
				use_nna = 1;
#else
				fprintf(stderr, "the NNA backend is in the T41 build only\n");
				return 2;
#endif
			} else if (strcmp(argv[ai + 1], "cpu")) {
				fprintf(stderr, "unknown engine %s\n", argv[ai + 1]);
				return 2;
			}
			ai += 2;
		} else
			break;
	}
	if (argc - ai != 4) {
		fprintf(stderr, "usage: %s [-e cpu|oracle|nna] [-c] model.bin model.graph input_rgba.raw prefix\n", argv[0]);
		return 2;
	}
	argv += ai - 1;
	if (nna_bin_load(&b, argv[1]) || nna_graph_load(&g, argv[2], &b))
		return 1;
	printf("bin: %d conv records, %d add records, %d header concat records; graph: %d ops, %d tensors\n",
	       b.nconv, b.nadd, b.nhcat, g.nnode, g.ntensor);
	T = calloc(g.ntensor, sizeof(*T));
	T[0].h = g.in_h, T[0].w = g.in_w, T[0].c = g.in_c;
	T[0].u8 = malloc((size_t)g.in_h * g.in_w * g.in_c);
	f = fopen(argv[3], "rb");
	if (!f || fread(T[0].u8, 1, (size_t)g.in_h * g.in_w * g.in_c, f) != (size_t)g.in_h * g.in_w * g.in_c) {
		fprintf(stderr, "cannot read %s\n", argv[3]);
		return 1;
	}
	fclose(f);
	t0 = now_ms();
	for (int k = 0; k < g.nnode; k++) {
		const struct nna_node *nd = &g.node[k];
		struct tensor *o = &T[nd->out], *x = &T[nd->in[0]];

		if (!x->u8) {
			fprintf(stderr, "op %d reads tensor %d before it is written\n", k, nd->in[0]);
			return 1;
		}
		switch (nd->op) {
		case OP_CONV: {
			int oh, ow;
			int32_t *S;
			const struct nna_conv_rec *r = &b.conv[nd->rec];

			if (x->c != nd->ic) {
				fprintf(stderr, "op %d: input has %d channels, conv wants %d\n", k, x->c, nd->ic);
				return 1;
			}
			if (use_nna) {
				S = malloc(sizeof(int32_t) * ((x->h + 2 * nd->pad) / nd->stride + 1) *
					   ((x->w + 2 * nd->pad) / nd->stride + 1) * nd->oc);
				if (nna_conv(&be, x->u8, x->h, x->w, x->c, nd, S, &oh, &ow))
					return 1;
				if (check) {
					int ch, cw;
					int32_t *Sc = conv_cpu(x, nd, &ch, &cw);
					long n = (long)oh * ow * nd->oc, d = 0;

					for (long i = 0; i < n; i++)
						d += S[i] != Sc[i];
					if (d || ch != oh || cw != ow) {
						printf("op %2d conv rec %2d %dx%d s%d %3d->%3d: %ld of %ld sums differ\n",
						       k, nd->rec, nd->kh, nd->kw, nd->stride, nd->ic, nd->oc, d, n);
						bad_layers++;
					}
					free(Sc);
				}
			} else {
				S = conv_cpu(x, nd, &oh, &ow);
			}
			o->h = oh, o->w = ow, o->c = nd->oc;
			if (nd->head) {
				o->f = malloc(sizeof(float) * oh * ow * nd->oc);
				nna_head(S, oh * ow, nd->oc, r->B, r->Fl, o->f);
				o->u8 = (uint8_t *)o->f;	/* marks it written */
			} else {
				o->u8 = malloc((size_t)oh * ow * nd->oc);
				nna_requant_silu(S, oh * ow, nd->oc, r->B, r->Fl, &r->silu, o->u8);
			}
			free(S);
			break;
		}
		case OP_ADD: {
			const struct nna_add_rec *a = &b.add[nd->arec];
			long n = (long)x->h * x->w * x->c;

			*o = *x;
			o->u8 = malloc(n);
			nna_add_ref(x->u8, T[nd->in[1]].u8, (int)n, a->wa, a->wb, a->B, a->T, o->u8);
			break;
		}
		case OP_CONCAT: {
			int C = 0, base = 0;
			long npx = (long)x->h * x->w;

			for (int i = 0; i < nd->nin; i++)
				C += T[nd->in[i]].c;
			o->h = x->h, o->w = x->w, o->c = C;
			o->u8 = malloc(npx * C);
			for (int i = 0; i < nd->nin; i++) {
				const struct tensor *xi = &T[nd->in[i]];
				uint8_t *src = xi->u8, *tmp = NULL;

				if (nd->a[i] != 1.0f) {
					tmp = malloc(npx * xi->c);
					nna_rescale(xi->u8, (int)(npx * xi->c), nd->a[i], tmp);
					src = tmp;
				}
				for (long p = 0; p < npx; p++)
					memcpy(&o->u8[p * C + base], &src[p * xi->c], xi->c);
				base += xi->c;
				free(tmp);
			}
			break;
		}
		case OP_MAXPOOL5:
			*o = *x;
			o->u8 = malloc((size_t)x->h * x->w * x->c);
			nna_maxpool5(x->u8, x->h, x->w, x->c, o->u8);
			break;
		case OP_UPSAMPLE2:
			o->h = 2 * x->h, o->w = 2 * x->w, o->c = x->c;
			o->u8 = malloc((size_t)o->h * o->w * o->c);
			nna_upsample2(x->u8, x->h, x->w, x->c, o->u8);
			break;
		}
	}
	printf("forward: %.1f ms (convs on %s)\n", now_ms() - t0, use_nna ? "the NNA" : "the CPU");
	if (check)
		printf("check: %d conv layers differ from the CPU conv\n", bad_layers);
	for (int i = 0; i < g.nout; i++) {
		char p[1024];
		const struct tensor *h = &T[g.out[i]];

		snprintf(p, sizeof(p), "%s%d.f32", argv[4], i);
		f = fopen(p, "wb");
		if (!f || fwrite(h->f, sizeof(float), (size_t)h->h * h->w * h->c, f) != (size_t)h->h * h->w * h->c)
			return 1;
		fclose(f);
		printf("%s: %dx%dx%d\n", p, h->h, h->w, h->c);
	}
	return 0;
}
