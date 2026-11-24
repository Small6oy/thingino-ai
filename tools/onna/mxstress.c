/* mxstress (stability test only): MXU3 vector loads + stores, no NN instructions, pinned to a CPU.
 *   mxstress [seconds] [cpu]   prints MB copied per second */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mxu3.h"

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
	enum { N = 1 << 20 };
	int secs = argc > 1 ? atoi(argv[1]) : 60, cpu = argc > 2 ? atoi(argv[2]) : 0;
	uint8_t *a, *b;
	cpu_set_t cs;

	CPU_ZERO(&cs);
	CPU_SET(cpu, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	setvbuf(stdout, NULL, _IONBF, 0);
	if (posix_memalign((void **)&a, 64, N) || posix_memalign((void **)&b, 64, N))
		return 1;
	for (int i = 0; i < N; i++)
		a[i] = (uint8_t)i;
	double t0 = now_s(), tl = t0;
	long mb = 0;

	for (;;) {
		for (size_t i = 0; i < N; i += 64) {
			VLOAD(1, a + i);
			VSTORE(1, b + i);
		}
		for (size_t i = 0; i < N; i += 64) {
			VLOAD(2, b + i);
			VSTORE(2, a + i);
		}
		mb += 2;
		double t = now_s();
		if (t - tl >= 1) {
			printf("%.0f s: %ld MB\n", t - t0, mb);
			tl = t;
		}
		if (t - t0 >= secs)
			break;
	}
	printf("done %ld MB\n", mb);
	return 0;
}
