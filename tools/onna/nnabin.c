/* nnabin: venus .bin + onna graph loader. See nnabin.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnabin.h"

#define ERR(...) do { fprintf(stderr, "nnabin: " __VA_ARGS__); fputc('\n', stderr); return -1; } while (0)

static int32_t rd32(const uint8_t *p)
{
	int32_t v;

	memcpy(&v, p, 4);
	return v;
}

static float rdf(const uint8_t *p)
{
	float v;

	memcpy(&v, p, 4);
	return v;
}

static long find(const struct nna_bin *b, long from, const void *pat, int n)
{
	for (long o = from < 0 ? 0 : from; o + n <= b->len; o++)
		if (!memcmp(b->data + o, pat, n))
			return o;
	return -1;
}

static const uint8_t SIG[10] = { 2, 0, 0, 0, 2, 0, 0, 0, 1, 0 };
static const uint8_t MARK[8] = { 4, 0, 0, 0, 0x20, 0, 0, 0 };

static int load_conv(struct nna_bin *b, long wend, struct nna_conv_rec *r)
{
	long ws = -1, mk;
	int n;
	int16_t raw[64];

	for (n = 1; n < 8192 && wend - 256L * n >= 36; n++) {
		const uint8_t *h = b->data + wend - 256L * n - 36;

		if (rd32(h) == 256 * n && rd32(h + 4) == 7 && rd32(h + 24) == 4 &&
		    rd32(h + 28) == 32 && rd32(h + 32) == 32) {
			ws = wend - 256L * n;
			r->nob = rd32(h + 8);
			r->kh = rd32(h + 16);
			r->kw = rd32(h + 20);
			break;
		}
	}
	if (ws < 0)
		ERR("no W header before the record at %ld", wend);
	if (r->nob < 1 || r->nob > 8 || r->kh < 1 || r->kh > 3 || r->kw < 1 || r->kw > 3)
		ERR("W header at %ld out of range (nob %d, %dx%d)", ws, r->nob, r->kh, r->kw);
	r->w = b->data + ws;
	r->nent = n;
	for (int k = 0; k < r->nob; k++)
		for (int i = 0; i < 32; i++) {
			r->B[32 * k + i] = rd32(b->data + wend + 0x25 + 256L * k + 4 * i);
			r->Fl[32 * k + i] = rdf(b->data + wend + 0xa5 + 256L * k + 4 * i);
		}
	/* the SiLU table: the marker within this record (heads have none), stored twice */
	mk = find(b, wend + 0x25 + 256L * r->nob - 0x80, MARK, 8);
	r->has_silu = mk >= 0 && mk < wend + 0x25 + 256L * r->nob + 4096 && mk + 8 + 256 <= b->len &&
		      !memcmp(b->data + mk + 8, b->data + mk + 8 + 128, 128);
	if (r->has_silu) {
		memcpy(raw, b->data + mk + 8, 128);
		nna_silu_load(&r->silu, raw);
		r->after = mk + 8 + 256;
	}
	return 0;
}

int nna_bin_load(struct nna_bin *b, const char *path)
{
	FILE *f = fopen(path, "rb");
	int cap = 0;
	long hcat_off[8];

	memset(b, 0, sizeof(*b));
	if (!f)
		ERR("cannot open %s", path);
	fseek(f, 0, SEEK_END);
	b->len = ftell(f);
	rewind(f);
	b->data = malloc(b->len);
	if (!b->data || fread(b->data, 1, b->len, f) != (size_t)b->len) {
		fclose(f);
		ERR("cannot read %s", path);
	}
	fclose(f);
	/* conv records: SIG with count 32 at +0x21 */
	for (long o = find(b, 0, SIG, 10); o >= 0; o = find(b, o + 1, SIG, 10)) {
		if (o + 0x25 > b->len || rd32(b->data + o + 0x21) != 32)
			continue;
		if (b->nconv == cap) {
			cap = cap ? 2 * cap : 64;
			b->conv = realloc(b->conv, cap * sizeof(*b->conv));
		}
		if (load_conv(b, o, &b->conv[b->nconv]))
			return -1;
		b->nconv++;
	}
	if (!b->nconv)
		ERR("no conv records");
	/* add records: 2, wA, wB, 2, sA, sB, B, T (header, first 64 KiB) */
	for (long o = 0; o + 32 <= b->len && o < 65536; o++) {
		const uint8_t *p = b->data + o;
		int wa = rd32(p + 4), wb = rd32(p + 8);
		float sa, sb, B, T;

		if (rd32(p) != 2 || rd32(p + 12) != 2 || wa <= 0 || wa > 255 || wb <= 0 || wb > 255 ||
		    (wa != 255 && wb != 255))
			continue;
		sa = rdf(p + 16), sb = rdf(p + 20), B = rdf(p + 24), T = rdf(p + 28);
		if (!(sa > 1e-4f && sa < 1 && sb > 1e-4f && sb < 1 && B < 0 && T > 1000 && T < 100000))
			continue;
		if (b->nadd == 16)
			ERR("too many add records");
		b->add[b->nadd++] = (struct nna_add_rec){ wa, wb, (int32_t)B, (int32_t)T };
		if ((int32_t)B != (int32_t)T / 2 - 8192 * (wa + wb))
			ERR("add record at %ld: B != T/2 - 8192(wA+wB)", o);
	}
	/* 2-input header concat records: 2, -128, -128, 2, s1, s2, 1/s_out */
	{
		const int32_t pat[4] = { 2, -128, -128, 2 };
		uint8_t pb[16];

		memcpy(pb, pat, 16);
		for (long o = find(b, 0, pb, 16); o >= 0; o = find(b, o + 1, pb, 16)) {
			float inv = rdf(b->data + o + 24);

			if (b->nhcat == 8)
				ERR("too many concat records");
			hcat_off[b->nhcat] = o;
			b->hcat[b->nhcat][0] = rdf(b->data + o + 16) * inv;
			b->hcat[b->nhcat][1] = rdf(b->data + o + 20) * inv;
			b->nhcat++;
		}
	}
	(void)hcat_off;
	return 0;
}

/* the concat factor in a type-30 record after conv record r */
static int type30(const struct nna_bin *b, int r, float *a)
{
	long o = b->conv[r].after, nb;

	if (!b->conv[r].has_silu || rd32(b->data + o) != 30)
		ERR("no type-30 record after conv record %d", r);
	nb = rd32(b->data + o + 148);
	o += 148 + 28 + nb;
	if (nb <= 0 || nb > 4096 || rd32(b->data + o) != 1 || rd32(b->data + o + 4) != 2 ||
	    rd32(b->data + o + 33) != 32)
		ERR("type-30 record after conv record %d: unexpected layout", r);
	*a = rdf(b->data + o + 37);
	for (long i = 1; i < nb / 4; i++)
		if (rdf(b->data + o + 37 + 4 * i) != *a)
			ERR("type-30 record after conv record %d: per-channel factors differ", r);
	return 0;
}

/* W groups (oc block -> input chunk -> taps raster) into [ky][kx][oc][ic] int8 */
static int build_w(const struct nna_conv_rec *r, struct nna_node *nd)
{
	int ngroups = r->nent / 4, g = 0;

	nd->wq = calloc((size_t)nd->kh * nd->kw * nd->oc * nd->ic, 1);
	for (int ob = 0; ob < (nd->oc + 31) / 32; ob++)
		for (int c = 0; c < (nd->stem ? 6 : nd->nch); c++)
			for (int t = 0; t < (nd->stem ? 1 : nd->kh * nd->kw); t++, g++) {
				if (g >= ngroups)
					ERR("conv record %d: %d groups, the graph needs more", nd->rec, ngroups);
				for (int o = 0; o < 32 && 32 * ob + o < nd->oc; o++)
					for (int i = 0; i < 32; i++) {
						int v = 0;

						for (int k = 0; k < 4; k++)
							v |= ((r->w[(4L * g + k) * 256 + o * 8 + i / 4] >> (2 * (i % 4))) & 3) << (2 * k);
						v -= 128;
						if (nd->stem) {		/* group = kx, ic = 4*ky + c */
							int ky = i / 4, cc = i % 4;

							if (i < 24)
								nd->wq[(((long)ky * 6 + c) * nd->oc + 32 * ob + o) * nd->ic + cc] = (int8_t)v;
						} else if (i < nd->chk[c]) {
							int ky = t / nd->kw, kx = t % nd->kw;

							nd->wq[(((long)ky * nd->kw + kx) * nd->oc + 32 * ob + o) * nd->ic + nd->ch0[c] + i] = (int8_t)v;
						}
					}
			}
	if (g != ngroups)
		ERR("conv record %d: %d groups, the graph uses %d", nd->rec, ngroups, g);
	return 0;
}

int nna_graph_load(struct nna_graph *g, const char *path, const struct nna_bin *b)
{
	FILE *f = fopen(path, "r");
	char line[1024];
	int cap = 0;

	memset(g, 0, sizeof(*g));
	if (!f)
		ERR("cannot open %s", path);
	while (fgets(line, sizeof(line), f)) {
		char *tok = strtok(line, " \n");
		struct nna_node nd;

		if (!tok)
			continue;
		memset(&nd, 0, sizeof(nd));
		if (!strcmp(tok, "onna-graph") || !strcmp(tok, "tensors")) {
			if (tok[0] == 't')
				g->ntensor = atoi(strtok(NULL, " \n"));
			continue;
		} else if (!strcmp(tok, "input")) {
			strtok(NULL, " ");
			g->in_h = atoi(strtok(NULL, " "));
			g->in_w = atoi(strtok(NULL, " "));
			g->in_c = atoi(strtok(NULL, " \n"));
			continue;
		} else if (!strcmp(tok, "output")) {
			while ((tok = strtok(NULL, " \n")) && g->nout < 4)
				g->out[g->nout++] = atoi(tok);
			continue;
		} else if (!strcmp(tok, "conv")) {
			int v[13];

			for (int i = 0; i < 13; i++)
				v[i] = atoi(strtok(NULL, " \n"));
			nd.op = OP_CONV;
			nd.out = v[0], nd.in[0] = v[1], nd.nin = 1, nd.rec = v[2], nd.oc = v[3], nd.ic = v[4];
			nd.kh = v[5], nd.kw = v[6], nd.stride = v[7], nd.pad = v[8], nd.padval = v[9];
			nd.stem = v[10], nd.head = v[11], nd.nch = v[12];
			for (int i = 0; i < nd.nch && i < 32; i++)
				if (sscanf(strtok(NULL, " \n"), "%d:%d", &nd.ch0[i], &nd.chk[i]) != 2)
					ERR("bad chunk in %s", path);
			if (nd.rec < 0 || nd.rec >= b->nconv)
				ERR("conv record %d not in the .bin", nd.rec);
			if (!nd.stem && (b->conv[nd.rec].kh != nd.kh || b->conv[nd.rec].kw != nd.kw ||
					 b->conv[nd.rec].nob != (nd.oc + 31) / 32))
				ERR("conv record %d does not match the graph's shape", nd.rec);
			if (!nd.head && !b->conv[nd.rec].has_silu)
				ERR("conv record %d has no SiLU table, but the graph applies SiLU", nd.rec);
			if (build_w(&b->conv[nd.rec], &nd))
				return -1;
		} else if (!strcmp(tok, "add")) {
			nd.op = OP_ADD;
			nd.out = atoi(strtok(NULL, " ")), nd.in[0] = atoi(strtok(NULL, " "));
			nd.in[1] = atoi(strtok(NULL, " ")), nd.nin = 2, nd.arec = atoi(strtok(NULL, " \n"));
			if (nd.arec < 0 || nd.arec >= b->nadd)
				ERR("add record %d not in the .bin", nd.arec);
		} else if (!strcmp(tok, "concat")) {
			nd.op = OP_CONCAT;
			nd.out = atoi(strtok(NULL, " ")), nd.nin = atoi(strtok(NULL, " "));
			if (nd.nin < 2 || nd.nin > 4)
				ERR("concat with %d inputs", nd.nin);
			for (int i = 0; i < nd.nin; i++)
				nd.in[i] = atoi(strtok(NULL, " \n"));
			for (int i = 0; i < nd.nin; i++) {
				char *s = strtok(NULL, " \n");
				int k, j;

				if (!s)
					ERR("concat without factor sources");
				if (!strcmp(s, "1"))
					nd.a[i] = 1.0f;
				else if (sscanf(s, "h%d.%d", &k, &j) == 2 && k < b->nhcat && j < 2)
					nd.a[i] = b->hcat[k][j];
				else if (sscanf(s, "t%d", &k) == 1 && k < b->nconv) {
					if (type30(b, k, &nd.a[i]))
						return -1;
				} else
					ERR("bad concat factor source '%s'", s);
			}
		} else if (!strcmp(tok, "maxpool5") || !strcmp(tok, "upsample2")) {
			nd.op = tok[0] == 'm' ? OP_MAXPOOL5 : OP_UPSAMPLE2;
			nd.out = atoi(strtok(NULL, " ")), nd.in[0] = atoi(strtok(NULL, " \n")), nd.nin = 1;
		} else {
			ERR("unknown op '%s'", tok);
		}
		if (g->nnode == cap) {
			cap = cap ? 2 * cap : 64;
			g->node = realloc(g->node, cap * sizeof(*g->node));
		}
		g->node[g->nnode++] = nd;
	}
	fclose(f);
	if (!g->ntensor || !g->nnode || !g->nout || !g->in_h)
		ERR("incomplete graph %s", path);
	return 0;
}
