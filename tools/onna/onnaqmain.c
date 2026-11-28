/*
 * onnaq command-line runner (usage and options: onnaq.c). It includes onnaq.c so that it can
 * drive the loader and the per-op timing directly; onnaq.c on its own is the library (onnaq.h).
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#include "onnaq.c"

int main(int argc, char **argv)
{
	int ai = 1, loops = 1;
	const char *eng = "cpu";

	while (ai < argc && argv[ai][0] == '-') {
		if (!strcmp(argv[ai], "-e") && ai + 1 < argc)
			eng = argv[++ai];
		else if (!strcmp(argv[ai], "-c"))
			check = 1;
		else if (!strcmp(argv[ai], "-T"))
			timing = 1;
		else if (!strcmp(argv[ai], "-TT"))
			timing = 2;
		else if (!strcmp(argv[ai], "-n") && ai + 1 < argc)
			loops = atoi(argv[++ai]);
		else
			break;
		ai++;
	}
	if (argc - ai != 4) {
		fprintf(stderr, "usage: %s [-e cpu|oracle|nna] [-c] [-T] [-n N] model.bin model.qgraph frame.rgb prefix\n", argv[0]);
		return 2;
	}
	if (!strcmp(eng, "oracle")) {
#ifdef ONNA_ORACLE
		nna_be_oracle(&be);
		use_nna = 1;
#else
		DIE("built without the oracle (make ORACLE=1)");
#endif
	} else if (!strcmp(eng, "nna")) {
#ifdef ONNA_MXU3
		cpu_set_t cs;

		CPU_ZERO(&cs);
		CPU_SET(0, &cs);
		if (sched_setaffinity(0, sizeof(cs), &cs))
			DIE("cannot pin to CPU0");
		nna_be_mxu3(&be);
		nna_pin_isp_cpu0();
		use_nna = 1;
#else
		DIE("not a T41 build");
#endif
	} else if (strcmp(eng, "cpu"))
		DIE("engine %s", eng);
	mallopt(M_MMAP_THRESHOLD, 64 << 20);	/* tensors from the heap, not fresh mmaps */
	mallopt(M_TRIM_THRESHOLD, 128 << 20);
	load_bin(argv[ai]);
	if (strcmp(argv[ai + 1], "-"))
		load_graph(argv[ai + 1]);
	else
		load_graph_bin(argv[ai]);		/* "-": the graph from the .bin's own op list */
	int nfold = getenv("ONNAQ_NOFOLD") ? 0 : fold();

	set_readers();
	plan_life();

	fprintf(stderr, "onnaq: %d W blocks, %d concat records, %d ops (%d folded into convs), input %dx%d\n", nrec,
		ncat, ng, nfold, in_w, in_h);

	size_t fsz = (size_t)in_h * in_w * in_c;	/* RGB, or RGBA (alpha 0) when the model takes 4 channels */
	uint8_t *rgb = xmalloc(fsz);
	FILE *f = fopen(argv[ai + 2], "rb");

	if (!f || fread(rgb, 1, fsz, f) != fsz)
		DIE("cannot read %d x %d x %d bytes from %s", in_w, in_h, in_c, argv[ai + 2]);
	fclose(f);

	/* every run's heads against run 0's, by a 64-bit FNV-1a of each head (no copy of the heads:
	 * on the camera a copy of yolov5n's first head alone is 3.9 MB) */
	uint64_t first[4] = { 0 };
	double best = 1e30, tot = 0;
	int pool0 = 0;		/* pool buffers after run 0: later runs must not add any */

	for (int it = 0; it < loops; it++) {
		double t0 = now_ms(), dt;

		run(rgb);
		dt = now_ms() - t0;
		if (!it)
			pool0 = npool;
		tot += dt;
		best = dt < best ? dt : best;
		for (int k = 0; k < nouts; k++) {
			const struct tensor *h = &T[outs[k]];
			const uint8_t *hb = (const uint8_t *)h->f;
			size_t n = (size_t)h->h * h->w * h->c * sizeof(float);
			uint64_t hv = 0xcbf29ce484222325ull;

			for (size_t i = 0; i < n; i++)
				hv = (hv ^ hb[i]) * 0x100000001b3ull;
			if (!it)
				first[k] = hv;
			else if (hv != first[k])
				DIE("run %d: head %d differs from run 0", it, k);
		}
	}
	for (int k = 0; k < nouts; k++) {
		const struct tensor *h = &T[outs[k]];
		char path[512];
		FILE *o;

		snprintf(path, sizeof(path), "%s.%d.f32", argv[ai + 3], k);
		o = fopen(path, "wb");
		if (!o || fwrite(h->f, sizeof(float), (size_t)h->h * h->w * h->c, o) != (size_t)h->h * h->w * h->c)
			DIE("cannot write %s", path);
		fclose(o);
		printf("head %d: %d x %d x %d -> %s\n", k, h->h, h->w, h->c, path);
	}
	if (timing) {
		static const char *nm[G_NOPS] = { "stem", "conv", "pc", "head", "maxpool2", "deconv2", "concat", "add",
						  "maxpool5", "upsample", "normalize" };

		for (int k = 0; k < G_NOPS; k++)
			printf("  %-9s %7.2f ms/run\n", nm[k], optime[k] / loops);
		if (timing > 1)		/* -TT: per node */
			for (int i = 0; i < ng; i++)
				if (!G[i].skip) {	/* a concat's input 0 from a folded deconv: that deconv's input */
					int t = G[i].dec[0] ? G[G[i].dec[0] - 1].in[0] : G[i].in[0];

					printf("  node %2d %-8s rec %2d %3dx%3dx%3d -> %3d  %6.2f ms\n", i, nm[G[i].op], G[i].rec,
					       T[t].h, T[t].w, T[t].c, T[G[i].out].c, nodetime[i] / loops);
				}
	}
	if (timing)
		printf("  tensors: %d pooled buffers, %.2f MB%s\n", npool, pool_bytes / 1e6,
		       npool != pool0 ? " (GREW after run 0)" : "");
	printf("%d run%s: %.2f ms avg, %.2f ms best (%s)%s\n", loops, loops > 1 ? "s" : "", tot / loops, best, eng,
	       loops > 1 ? ", heads identical in every run" : "");
	return 0;
}
