/* haltest: drive raptor-hal's hal_jzdl_* API on NV12 frames and print every
 * detection exactly (hex floats), so the libjzdl backend (hal_ivs_jzdl.cpp)
 * and the ojzdl backend (hal_ivs_ojzdl.c) can be compared byte for byte.
 *   haltest MODEL.bin W H NCLASSES CONF NMS RUNS FRAME.nv12 [FRAME.nv12 ...]
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "raptor_hal.h"

static void logf_(int level, const char *file, int line, const char *fmt, ...)
{
	va_list ap;
	(void)file; (void)line;
	fprintf(stderr, "[hal %d] ", level);
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
}
rss_hal_log_func_t rss_hal_log_fn = logf_;

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
	rss_ivs_jzdl_param_t p = {0};
	int w, h, runs, i, f;
	void *ctx;
	if (argc < 9) {
		fprintf(stderr, "usage: %s MODEL W H NCLASSES CONF NMS RUNS FRAME.nv12...\n", argv[0]);
		return 2;
	}
	snprintf(p.model_path, sizeof(p.model_path), "%s", argv[1]);
	w = p.width = atoi(argv[2]);
	h = p.height = atoi(argv[3]);
	p.num_classes = atoi(argv[4]);
	p.conf_threshold = (float)atof(argv[5]);
	p.nms_threshold = (float)atof(argv[6]);
	runs = atoi(argv[7]);
	ctx = hal_jzdl_create(&p);
	if (!ctx)
		return 1;
	for (f = 8; f < argc; f++) {
		size_t n = (size_t)w * h * 3 / 2;
		unsigned char *buf = malloc(n);
		FILE *fp = fopen(argv[f], "rb");
		double best = 1e30;
		rss_ivs_detect_result_t r;
		if (!buf || !fp || fread(buf, 1, n, fp) != n)
			return 1;
		fclose(fp);
		for (i = 0; i < runs; i++) {
			double t0 = now_ms(), dt;
			if (hal_jzdl_detect(ctx, buf, &r) != 0)
				return 1;
			dt = now_ms() - t0;
			if (dt < best)
				best = dt;
		}
		printf("frame %s: %d detections, detect min %.1f ms\n", argv[f], r.count, best);
		for (i = 0; i < r.count; i++)
			printf("  cls %d conf %a (%.4f) box %d %d %d %d\n", r.detections[i].class_id,
			       (double)r.detections[i].confidence, (double)r.detections[i].confidence,
			       r.detections[i].box.p0_x, r.detections[i].box.p0_y, r.detections[i].box.p1_x,
			       r.detections[i].box.p1_y);
		free(buf);
	}
	hal_jzdl_destroy(ctx);
	return 0;
}
