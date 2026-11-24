// jzdlbench: time libjzdl's forward (input + run) N times on one frame, without
// the model load. Counterpart of `ojzdlrun -n N` for step-2 comparisons.
//
//   jzdlbench MODEL.bin INPUT.rgb RUNS
//
// Build like jzdlrun (README.md). Runs under QEMU or on the v3.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <time.h>
#include "net.h"
#include "utils.h"

static uint64_t now_ns()
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: %s MODEL.bin INPUT.rgb RUNS\n", argv[0]);
		return 2;
	}
	int runs = atoi(argv[3]);
	uint64_t t0 = now_ns();
	jzdl::BaseNet *net = jzdl::net_create();
	if (!net || net->load_model(argv[1]) != 0) {
		fprintf(stderr, "cannot load %s\n", argv[1]);
		return 1;
	}
	uint64_t tl = now_ns() - t0;
	std::vector<uint32_t> s = net->get_input_shape();
	int w = s[0], h = s[1], c = s[2];
	std::vector<uint8_t> rgb((size_t)w * h * c), work(rgb.size());
	FILE *f = fopen(argv[2], "rb");
	if (!f || fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
		fprintf(stderr, "cannot read %s\n", argv[2]);
		return 1;
	}
	fclose(f);
	uint64_t best = ~0ull, total = 0;
	for (int r = 0; r < runs; r++) {
		work = rgb;
		uint64_t a = now_ns();
		jzdl::Mat<uint8_t> img(w, h, c, work.data());
		jzdl::image_sub(img, 128);
		jzdl::Mat<int8_t> in(w, h, c, (int8_t *)img.data);
		jzdl::Mat<float> out;
		if (net->input(in) != 0 || net->run(out, -999) != 0) {
			fprintf(stderr, "run failed\n");
			return 1;
		}
		uint64_t dt = now_ns() - a;
		total += dt;
		if (dt < best)
			best = dt;
	}
	printf("libjzdl load %.3f ms; forward: %d runs, min %.3f ms, avg %.3f ms\n",
	       tl / 1e6, runs, best / 1e6, total / 1e6 / runs);
	jzdl::net_destory(net);
	return 0;
}
