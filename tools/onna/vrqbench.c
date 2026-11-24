/* vrqbench: cost of the MXU3 requant per pixel (32 channels) on the camera (test only) */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "nnacpu.h"

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void)
{
	static struct nna_vrq v;
	static int32_t S[2][16] __attribute__((aligned(64)));
	static uint8_t q[64] __attribute__((aligned(64)));
	enum { N = 1000000 };
	cpu_set_t cs;
	double t;

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	for (int c = 0; c < 16; c++) {
		v.B[0][c] = v.B[1][c] = c;
		v.Fl[0][c] = v.Fl[1][c] = 0.5f;
		S[0][c] = S[1][c] = 1000 * c;
	}
	for (int i = 0; i < 32; i++)
		v.tb.y[i] = (int16_t)(i * 1000), v.tb.dy[i] = 1000;
	nna_vrq_load(&v);
	t = now_ms();
	for (int i = 0; i < N; i++)
		nna_vrq_px(S[0], S[1], q);
	t = now_ms() - t;
	printf("nna_vrq_px: %.1f ns per pixel (32 ch)\n", t * 1e6 / N);
	t = now_ms();
	for (int i = 0; i < N / 10; i++)
		nna_vrq_load(&v);
	t = now_ms() - t;
	printf("nna_vrq_load: %.1f ns\n", t * 1e6 / (N / 10));
	t = now_ms();
	for (int i = 0; i < N; i++)
		nna_requant_t(S[0], 2, v.B[0], v.Fl[0], (int32_t *)q);
	t = now_ms() - t;
	printf("t only (2 vectors): %.1f ns\n", t * 1e6 / N);
	return q[0] == 255;
}
