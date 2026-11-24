/*
 * onnarun: run a venus .bin with onna. Writes <prefix>0.f32 .. for the heads.
 *   onnarun [-e cpu|oracle|nna] [-c] model.bin model.graph input_rgba.raw prefix
 *   -e  where convs run: cpu (plain integer loop), oracle (the NNA driver against the host
 *       test oracle; builds with ORACLE=1) or nna (the real NNA; T41 build, pins itself to
 *       CPU0). Default cpu.
 *   -c  also run the CPU conv and compare every layer's sums.
 *   -l N  stop after op N (with -c: debug the first layers on a camera).
 *   -t  per-op times and NN instruction counts.
 *   -n N  run the forward pass N times (tables built once at load) and print min/avg.
 * prefix "-": print a 64-bit FNV-1a of each head instead of writing files (camera: /tmp is RAM).
 * Tensors are freed after their last use.
 */
#ifdef ONNA_MXU3
#define _GNU_SOURCE		/* sched_setaffinity: the NN unit is on CPU0 only */
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <malloc.h>
#endif
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
	int ps;		/* bytes per pixel in u8 when not c (0 = c): 32 for the NNA's 16-channel outputs */
	int view;	/* u8 points into another tensor's buffer (a concat's): never freed on its own */
	uint8_t *u8;
	float *f;
};

static uint64_t fnv64(const void *p, size_t n)
{
	const uint8_t *b = p;
	uint64_t h = 0xcbf29ce484222325ull;

	for (size_t i = 0; i < n; i++)
		h = (h ^ b[i]) * 0x100000001b3ull;
	return h;
}

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

/* tensor memory: 64-byte aligned, so the MXU3 paths (32-byte aligned loads/stores) always apply */
static void *amalloc(size_t n)
{
	void *p = NULL;

	return posix_memalign(&p, 64, n ? n : 64) ? NULL : p;
}

#define PS(t) ((t)->ps ? (t)->ps : (t)->c)

/* a strided tensor (ps > c) in place as a dense one */
static int dense(struct tensor *t)
{
	uint8_t *d;

	if (!t->ps || t->ps == t->c)
		return 0;
	d = amalloc((size_t)t->h * t->w * t->c);
	if (!d)
		return -1;
	for (long p = 0; p < (long)t->h * t->w; p++)
		memcpy(d + p * t->c, t->u8 + p * t->ps, (size_t)t->c);
	if (!t->view)
		free(t->u8);
	t->u8 = d;
	t->ps = 0;
	t->view = 0;
	return 0;
}

/* npx pixels of c bytes from src (stride sps) to dst (stride C): sized copies the compiler inlines */
static void copy_px(uint8_t *dst, int C, const uint8_t *src, int sps, int c, long npx)
{
#define COPY_PX(n) for (long p = 0; p < npx; p++, dst += C, src += sps) memcpy(dst, src, (n)); return
	switch (c) {
	case 16: COPY_PX(16);
	case 32: COPY_PX(32);
	case 64: COPY_PX(64);
	case 128: COPY_PX(128);
	case 256: COPY_PX(256);
	default: COPY_PX((size_t)c);
	}
#undef COPY_PX
}

#ifdef ONNA_MXU3
/* a conv's numbers for the MXU3 SiLU requant, per 32-oc block, built on first use */
static const struct nna_vrq *vrq_of(const struct nna_conv_rec *r, const struct nna_node *nd)
{
	static struct nna_vrq *cache[256];
	int nob = (nd->oc + 31) / 32;
	void *p = NULL;

	if ((!r->has_silu && !nd->head) || nd->rec < 0 || nd->rec >= 256)
		return NULL;
	if (cache[nd->rec])
		return cache[nd->rec];
	if (posix_memalign(&p, 64, sizeof(struct nna_vrq) * (size_t)nob))
		return NULL;
	cache[nd->rec] = p;
	for (int ob = 0; ob < nob; ob++) {
		struct nna_vrq *v = &cache[nd->rec][ob];

		memset(v, 0, sizeof(*v));
		for (int h = 0; h < 2; h++)
			for (int c = 0; c < 16; c++)
				if (32 * ob + 16 * h + c < nd->oc) {
					v->B[h][c] = r->B[32 * ob + 16 * h + c];
					v->Fl[h][c] = r->Fl[32 * ob + 16 * h + c];
				}
		if (nd->oc - 32 * ob == 16)	/* paired pixels use half 0's numbers twice */
			for (int c = 0; c < 16; c++)
				v->B[1][c] = v->B[0][c], v->Fl[1][c] = v->Fl[0][c];
		if (r->has_silu)
			v->tb = r->silu;
		v->wide = nd->oc == 16;		/* written as 32-byte pixels (see the fused conv) */
	}
	return cache[nd->rec];
}
#endif

int main(int argc, char **argv)
{
	struct nna_bin b;
	struct nna_graph g;
	struct tensor *T;
	FILE *f;
	double t0;

	struct nna_be be = { 0 };
	int use_nna = 0, check = 0, ai = 1, bad_layers = 0, limit = 1 << 30, timing = 0, runs = 1;

#ifdef __linux__
	/* keep freed tensors in the heap: big blocks would otherwise be mmap'd per frame and
	 * every output page faulted in again (allocations repeat exactly from frame to frame) */
	mallopt(M_MMAP_THRESHOLD, 64 << 20);
	mallopt(M_TRIM_THRESHOLD, 128 << 20);
#endif

	while (ai < argc && argv[ai][0] == '-') {
		if (!strcmp(argv[ai], "-c"))
			check = 1, ai++;
		else if (!strcmp(argv[ai], "-n") && ai + 1 < argc)
			runs = atoi(argv[ai + 1]), ai += 2;
		else if (!strcmp(argv[ai], "-t"))
			timing = 1, ai++;
		else if (!strcmp(argv[ai], "-T"))	/* per-op times only: no counting wrapper */
			timing = 2, ai++;
		else if (!strcmp(argv[ai], "-l") && ai + 1 < argc)
			limit = atoi(argv[ai + 1]), ai += 2;
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
		fprintf(stderr, "usage: %s [-e cpu|oracle|nna] [-c] [-t|-T] [-n N] [-l N] model.bin model.graph input_rgba.raw prefix\n", argv[0]);
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
	static uint8_t *add_lut[16];

	for (int k = 0; k < g.nnode; k++)	/* tables: built once, outside the timed forward */
		if (g.node[k].op == OP_ADD && !add_lut[g.node[k].arec]) {
			const struct nna_add_rec *a = &b.add[g.node[k].arec];

			add_lut[g.node[k].arec] = malloc(65536);
			nna_add_lut(a->wa, a->wb, a->B, a->T, add_lut[g.node[k].arec]);
		}

	/* last use of every tensor (outputs: never freed) */
	int *last = calloc(g.ntensor, sizeof(int));

	for (int k = 0; k < g.nnode; k++)
		for (int i = 0; i < g.node[k].nin; i++)
			last[g.node[k].in[i]] = k;
	for (int i = 0; i < g.nout; i++)
		last[g.out[i]] = g.nnode;
	/* concat outputs written in place: a fused conv's output that is a factor-1 concat input goes
	 * straight into the concat's buffer (pixel stride = the concat's channels) when its offset is
	 * 32-byte aligned, it isn't a 16-channel output, all its other readers are convs, and the
	 * concat's buffer outlives it */
	int *vcat = malloc(sizeof(int) * g.ntensor), *vbase = malloc(sizeof(int) * g.ntensor);
	int *prod = malloc(sizeof(int) * g.ntensor), nviews = 0;

	for (int i = 0; i < g.ntensor; i++)
		vcat[i] = -1, prod[i] = -1;
	int *tch = calloc(g.ntensor, sizeof(int));	/* channels of every tensor */

	tch[0] = g.in_c;
	for (int k = 0; k < g.nnode; k++) {
		const struct nna_node *n = &g.node[k];

		prod[n->out] = k;
		if (n->op == OP_CONV)
			tch[n->out] = n->oc;
		else if (n->op == OP_CONCAT)
			for (int i = 0; i < n->nin; i++)
				tch[n->out] += tch[n->in[i]];
		else
			tch[n->out] = tch[n->in[0]];
	}
	for (int k = 0; k < g.nnode && use_nna && !check && !getenv("ONNA_NOVIEW"); k++) {
		const struct nna_node *cn = &g.node[k];
		int C = 0, base = 0;

		if (cn->op != OP_CONCAT)
			continue;
		C = tch[cn->out];
		for (int i = 0; i < cn->nin; i++) {
			int ti = cn->in[i], pk = prod[ti], ok = pk >= 0;
			const struct nna_node *pn = ok ? &g.node[pk] : NULL;

			if (ok)
				ok = pn->op == OP_CONV && !pn->head && pn->oc % 32 == 0 && pn->oc != 16 &&
				     cn->a[i] == 1.0f && C > 0 && C % 32 == 0 && base % 32 == 0 &&
				     last[cn->out] >= last[ti] && vcat[ti] < 0;
			for (int r = 0; r < g.nnode && ok; r++)
				for (int j = 0; j < g.node[r].nin; j++)
					if (g.node[r].in[j] == ti && r != k && g.node[r].op != OP_CONV)
						ok = 0;
			if (ok)
				vcat[ti] = cn->out, vbase[ti] = base, nviews++;
			base += tch[ti];
		}
	}
	if (nviews && timing)
		printf("concat inputs written in place: %d\n", nviews);
	if (use_nna && timing == 1)
		nna_count_wrap(&be);	/* count instructions: wrap once, not per run */
	double best = 1e30, sum = 0;
	uint8_t *in0 = T[0].u8;
	struct tensor *T0 = malloc(sizeof(*T) * g.ntensor);

	memcpy(T0, T, sizeof(*T) * g.ntensor);
	uint64_t head0[8] = { 0 };
	int run_diff = 0;

	for (int run = 0; run < runs; run++) {
	if (run) {		/* fresh tensors, same input */
		for (int i = 1; i < g.ntensor; i++)
			if (T[i].u8 && last[i] == g.nnode)
				free(T[i].u8);
		memcpy(T, T0, sizeof(*T) * g.ntensor);
		T[0].u8 = in0;
	}
	t0 = now_ms();
	double t_kind[5] = { 0 }, t_nna = 0, t_rq = 0;
	long cnt0[5], cnt1[5];
	static const char *kname[5] = { "conv", "add", "concat", "maxpool", "upsample" };

	for (int k = 0; k < g.nnode && k <= limit; k++) {
		const struct nna_node *nd = &g.node[k];
		double tk = now_ms();

		if (timing)
			nna_count_get(cnt0);
		struct tensor *o = &T[nd->out], *x = &T[nd->in[0]];

		if (!x->u8) {
			fprintf(stderr, "op %d reads tensor %d before it is written\n", k, nd->in[0]);
			return 1;
		}
		o->ps = 0;
		o->view = 0;
		/* only conv, add and concat take strided pixels; others (and the CPU conv) get them dense */
		if (nd->op == OP_MAXPOOL5 || nd->op == OP_UPSAMPLE2 || (nd->op == OP_CONV && (!use_nna || check)))
			for (int i = 0; i < nd->nin; i++)
				if (dense(&T[nd->in[i]]))
					return 1;
		if (nd->op == OP_ADD && PS(x) != PS(&T[nd->in[1]]) && (dense(x) || dense(&T[nd->in[1]])))
			return 1;
		switch (nd->op) {
		case OP_CONV: {
			int oh, ow;
			int32_t *S;
			const struct nna_conv_rec *r = &b.conv[nd->rec];

			if (x->c != nd->ic) {
				fprintf(stderr, "op %d: input has %d channels, conv wants %d\n", k, x->c, nd->ic);
				return 1;
			}
			if (use_nna && !check && !nd->head && nd->oc % 16 == 0) {
				/* fused: sums are requantized as they are read back; no S buffer */
				int eh = (x->h + 2 * nd->pad - nd->kh) / nd->stride + 1;
				int ew = (x->w + 2 * nd->pad - nd->kw) / nd->stride + 1;
				struct nna_rq rq = { r->B, r->Fl, r->lut, NULL, NULL, NULL, nd->oc == 16 ? 32 : 0 };
				void *qa = NULL;

				if (vcat[nd->out] >= 0) {	/* into the concat's buffer */
					struct tensor *ct = &T[vcat[nd->out]];
					int C = tch[vcat[nd->out]];
					if (!ct->u8) {
						ct->u8 = amalloc((size_t)eh * ew * C);
						if (!ct->u8)
							return 1;
						ct->h = eh, ct->w = ew, ct->c = C;
					}
					rq.q = ct->u8 + vbase[nd->out];
					rq.ps = C;
				} else {
					if (posix_memalign(&qa, 64, (size_t)eh * ew * (nd->oc == 16 ? 32 : nd->oc)))
						return 1;
					rq.q = qa;
				}
#ifdef ONNA_MXU3
				rq.vrq = vrq_of(r, nd);
#endif
				double ta = now_ms();

				if (nna_conv(&be, x->u8, x->h, x->w, x->c, PS(x), nd, NULL, &rq, &oh, &ow))
					return 1;
				t_nna += now_ms() - ta;
				o->h = oh, o->w = ow, o->c = nd->oc, o->ps = rq.ps;
				o->u8 = rq.q;
				o->view = vcat[nd->out] >= 0;
				break;
			}
			if (use_nna && !check && nd->head) {
				/* fused head: floats straight from the read-back; no S buffer */
				int eh = (x->h + 2 * nd->pad - nd->kh) / nd->stride + 1;
				int ew = (x->w + 2 * nd->pad - nd->kw) / nd->stride + 1;
				struct nna_rq rq = { r->B, r->Fl, NULL, NULL, NULL, NULL, 0 };
				void *fa = NULL;

				if (posix_memalign(&fa, 64, sizeof(float) * (size_t)eh * ew * nd->oc))
					return 1;
				rq.f = fa;
#ifdef ONNA_MXU3
				rq.vrq = vrq_of(r, nd);
#endif
				double ta = now_ms();

				if (nna_conv(&be, x->u8, x->h, x->w, x->c, PS(x), nd, NULL, &rq, &oh, &ow))
					return 1;
				t_nna += now_ms() - ta;
				o->h = oh, o->w = ow, o->c = nd->oc;
				o->f = rq.f;
				o->u8 = (uint8_t *)o->f;	/* marks it written */
				break;
			}
			if (use_nna) {
				S = malloc(sizeof(int32_t) * ((x->h + 2 * nd->pad) / nd->stride + 1) *
					   ((x->w + 2 * nd->pad) / nd->stride + 1) * nd->oc);
				double ta = now_ms();

				if (nna_conv(&be, x->u8, x->h, x->w, x->c, PS(x), nd, S, NULL, &oh, &ow))
					return 1;
				t_nna += now_ms() - ta;
				if (check) {
					int ch, cw;
					int32_t *Sc = conv_cpu(x, nd, &ch, &cw);
					long n = (long)oh * ow * nd->oc, d = 0;

					for (long i = 0; i < n; i++)
						d += S[i] != Sc[i];
					if (d || ch != oh || cw != ow) {
						printf("op %2d conv rec %2d %dx%d s%d %3d->%3d: %ld of %ld sums differ\n",
						       k, nd->rec, nd->kh, nd->kw, nd->stride, nd->ic, nd->oc, d, n);
						if (getenv("ONNA_TILEDUMP")) {	/* first 2x4 tile, two channels */
							for (int c = 0; c < 2; c++)
								for (int yy = 0; yy < 2; yy++) {
									printf("   oc%d row%d nna:", c, yy);
									for (int xx = 0; xx < 4; xx++)
										printf(" %7d", S[((long)yy * ow + xx) * nd->oc + c]);
									printf("   cpu:");
									for (int xx = 0; xx < 4; xx++)
										printf(" %7d", Sc[((long)yy * ow + xx) * nd->oc + c]);
									printf("\n");
								}
						}
						if (!bad_layers)
							for (long i = 0, shown = 0; i < n && shown < 6; i++)
								if (S[i] != Sc[i]) {
									printf("   [px %ld oc %ld] nna %d cpu %d\n", i / nd->oc,
									       i % nd->oc, S[i], Sc[i]);
									shown++;
								}
						bad_layers++;
					} else {
						printf("op %2d conv rec %2d %dx%d s%d %3d->%3d: all %ld sums equal\n",
						       k, nd->rec, nd->kh, nd->kw, nd->stride, nd->ic, nd->oc, n);
					}
					free(Sc);
				}
			} else {
				S = conv_cpu(x, nd, &oh, &ow);
			}
			o->h = oh, o->w = ow, o->c = nd->oc;
			if (nd->head) {
				o->f = amalloc(sizeof(float) * oh * ow * nd->oc);
				nna_head(S, oh * ow, nd->oc, r->B, r->Fl, o->f);
				o->u8 = (uint8_t *)o->f;	/* marks it written */
			} else {
				double tb = now_ms();

				o->u8 = amalloc((size_t)oh * ow * nd->oc);
				nna_requant_lut(S, oh * ow, nd->oc, r->B, r->Fl, r->lut, o->u8);
				t_rq += now_ms() - tb;
			}
			free(S);
			break;
		}
		case OP_ADD: {
			/* elementwise over the stored bytes (strided inputs: the padding channels too) */
			long n = (long)x->h * x->w * PS(x);

			void *pa = NULL;

			*o = *x;
			if (posix_memalign(&pa, 64, (size_t)n))
				return 1;
			o->u8 = pa;
#ifdef ONNA_MXU3
			if (!(((uintptr_t)x->u8 | (uintptr_t)T[nd->in[1]].u8) & 31)) {
				const struct nna_add_rec *ad = &b.add[nd->arec];

				nna_vadd(x->u8, T[nd->in[1]].u8, (int)n, ad->wa, ad->wb, ad->B, ad->T, o->u8);
				break;
			}
#endif
			nna_add_apply(x->u8, T[nd->in[1]].u8, (int)n, add_lut[nd->arec], o->u8);
			break;
		}
		case OP_CONCAT: {
			int C = 0, base = 0;
			long npx = (long)x->h * x->w;

			for (int i = 0; i < nd->nin; i++)
				C += T[nd->in[i]].c;
			o->h = x->h, o->w = x->w, o->c = C;
			if (!o->u8)		/* else a producer allocated it (inputs written in place) */
				o->u8 = amalloc(npx * C);
#ifdef ONNA_MXU3
			if (nd->nin == 2 && C == 32 && !(npx & 1) && nd->a[0] == 1.0f && nd->a[1] == 1.0f &&
			    PS(x) == 32 && PS(&T[nd->in[1]]) == 32 && x->c == 16) {
				/* two 16-channel tensors in 32-byte pixels: shuffle + OR */
				nna_vmerge16(x->u8, T[nd->in[1]].u8, npx, o->u8);
				break;
			}
#endif
			for (int i = 0; i < nd->nin; i++) {
				const struct tensor *xi = &T[nd->in[i]];
				const uint8_t *src = xi->u8;
				int c = xi->c, sps = PS(xi);

				if (src == o->u8 + base && sps == C) {	/* already in place */
					base += c;
					continue;
				}

#ifdef ONNA_MXU3
				if (nd->a[i] != 1.0f && !(c & 63) && !(sps & 63) && !(C & 63) && !(base & 63)) {
					/* rescaled on MXU3 straight into place */
					nna_vrescale_px(src, sps, &o->u8[base], C, c, npx, nd->a[i]);
					base += c;
					continue;
				}
				if (nd->a[i] != 1.0f && !((uintptr_t)src & 31)) {
					/* rescaled on MXU3 into a packed buffer, then copied into place */
					void *ra = NULL;

					if (posix_memalign(&ra, 64, (size_t)npx * sps))
						return 1;
					nna_vrescale(src, (int)(npx * sps), nd->a[i], ra);
					copy_px(&o->u8[base], C, ra, sps, c, npx);
					free(ra);
					base += c;
					continue;
				}
#endif
				if (nd->a[i] != 1.0f) {
					/* the rescale is a function of the byte: a 256-entry table, built
					 * with nna_rescale itself (exact), applied while copying */
					uint8_t id[256], map[256];

					for (int v = 0; v < 256; v++)
						id[v] = (uint8_t)v;
					nna_rescale(id, 256, nd->a[i], map);
					for (long p = 0; p < npx; p++) {
						uint8_t *d = &o->u8[p * C + base];
						const uint8_t *s = &src[p * sps];

						for (int j = 0; j < c; j++)
							d[j] = map[s[j]];
					}
				} else {
					copy_px(&o->u8[base], C, src, sps, c, npx);
				}
				base += c;
			}
			break;
		}
		case OP_MAXPOOL5: {
			void *pa = NULL;

			*o = *x;
			if (posix_memalign(&pa, 64, (size_t)x->h * x->w * x->c))
				return 1;
			o->u8 = pa;
			nna_maxpool5(x->u8, x->h, x->w, x->c, o->u8);
			break;
		}
		case OP_UPSAMPLE2:
			o->h = 2 * x->h, o->w = 2 * x->w, o->c = x->c;
			o->u8 = amalloc((size_t)o->h * o->w * o->c);
			nna_upsample2(x->u8, x->h, x->w, x->c, o->u8);
			break;
		}
		if (timing) {
			double dt = now_ms() - tk;

			t_kind[nd->op] += dt;
			if (nd->op == OP_CONV) {
				nna_count_get(cnt1);
				printf("  op %2d conv %dx%d s%d %3d->%3d %3dx%3d: %7.1f ms  rwr %ld dwr %ld drd %ld mac %ld cmd %ld\n",
				       k, nd->kh, nd->kw, nd->stride, nd->ic, nd->oc, o->h, o->w, dt,
				       cnt1[0] - cnt0[0], cnt1[1] - cnt0[1], cnt1[2] - cnt0[2], cnt1[3] - cnt0[3], cnt1[4] - cnt0[4]);
			} else {
				printf("  op %2d %-8s %3dx%3dx%3d: %7.2f ms\n", k, kname[nd->op], o->h, o->w, o->c, dt);
			}
		}
		for (int i = 0; i < nd->nin; i++) {
			struct tensor *ti = &T[nd->in[i]];

			if (last[nd->in[i]] == k && ti->u8 && nd->in[i] != 0) {
				if (!ti->view)
					free(ti->u8);
				ti->u8 = NULL;
				ti->f = NULL;
			}
		}
	}
	double dt_run = now_ms() - t0;

	/* every run must give the same heads (a lost vector register would show here) */
	for (int i = 0; i < g.nout; i++) {
		const struct tensor *h = &T[g.out[i]];
		uint64_t hv = h->f ? fnv64(h->f, sizeof(float) * h->h * h->w * h->c) : 0;

		if (!run)
			head0[i] = hv;
		else if (hv != head0[i])
			run_diff++;
	}

	best = dt_run < best ? dt_run : best;
	sum += dt_run;
	printf("forward: %.1f ms (convs on %s)\n", dt_run, use_nna ? "the NNA" : "the CPU");
	if (timing)
		for (int i = 0; i < 5; i++)
			printf("  total %-8s %8.1f ms\n", kname[i], t_kind[i]);
	if (timing)
		printf("  conv split: NNA driver %.1f ms, requant+SiLU %.1f ms\n", t_nna, t_rq);
	if (check)
		printf("check: %d conv layers differ from the CPU conv\n", bad_layers);
	}
	if (runs > 1)
		printf("forward over %d runs: min %.1f ms, avg %.1f ms; heads %s in every run\n", runs, best,
		       sum / runs, run_diff ? "DIFFER" : "identical");
	for (int i = 0; i < g.nout && limit >= g.nnode; i++) {
		char p[1024];
		const struct tensor *h = &T[g.out[i]];

		if (!strcmp(argv[4], "-")) {
			printf("head %d: %dx%dx%d fnv64 %016llx\n", i, h->h, h->w, h->c,
			       (unsigned long long)fnv64(h->f, sizeof(float) * h->h * h->w * h->c));
			continue;
		}
		snprintf(p, sizeof(p), "%s%d.f32", argv[4], i);
		f = fopen(p, "wb");
		if (!f || fwrite(h->f, sizeof(float), (size_t)h->h * h->w * h->c, f) != (size_t)h->h * h->w * h->c)
			return 1;
		fclose(f);
		printf("%s: %dx%dx%d\n", p, h->h, h->w, h->c);
	}
	return 0;
}
