/* membench (test only): the v4's data cache line size, and whether PrepareForStore (pref 30)
 * speeds up streaming vector stores */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mxu3.h"

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define PREP(p) __asm__ volatile("pref 30, 0(%0)" :: "r"(p) : "memory")
#define FLUSHALL(buf, n) do { for (size_t i_ = 0; i_ < (n); i_ += 32) __asm__ volatile("cache 0x15, 0(%0)" :: "r"((buf) + i_)); } while (0)

int main(void)
{
	enum { N = 8 << 20 };
	uint8_t *buf, *big;
	static uint8_t v[64] __attribute__((aligned(64)));
	cpu_set_t cs;
	double t;

	CPU_ZERO(&cs);
	CPU_SET(0, &cs);
	sched_setaffinity(0, sizeof(cs), &cs);
	if (posix_memalign((void **)&buf, 64, 4096) || posix_memalign((void **)&big, 64, N))
		return 1;
	/* line size: fill, evict (by streaming through big), PrepareForStore at A, write 32 bytes,
	 * then see whether bytes 32..63 kept their value */
	for (int trial = 0; trial < 3; trial++) {
		memset(buf, 0x11, 4096);
		memset(big, 0x5a, N);		/* evicts buf from L1 and L2 */
		uint8_t *A = buf + 1024;
		PREP(A);
		memset(A, 0x22, 32);
		printf("after pref 30 + 32-byte store: A[32..35] = %02x %02x %02x %02x, A[-4] = %02x\n",
		       A[32], A[33], A[34], A[35], A[-1]);
	}
	memset(v, 7, 64);
	VLOAD(1, v);
	for (int pass = 0; pass < 4; pass++) {
		int prep = pass & 1;

		memset(big, 0, N);		/* touch pages */
		memset(big, 1, N);
		t = now_ms();
		for (size_t i = 0; i < N; i += 64) {
			if (prep) {
				PREP(big + i);
				PREP(big + i + 32);
			}
			VSTORE(1, big + i);
		}
		t = now_ms() - t;
		printf("vector store 8 MB %s: %.2f ms (%.0f MB/s)\n", prep ? "with pref 30" : "plain       ", t, 8.0 / t * 1000);
	}
	for (int pass = 0; pass < 2; pass++) {
		volatile uint32_t sum = 0;
		t = now_ms();
		for (size_t i = 0; i < N; i += 32)
			sum += *(volatile uint32_t *)(big + i);
		t = now_ms() - t;
		printf("read 8 MB, one word per 32 B: %.2f ms (%.0f MB/s)\n", t, 8.0 / t * 1000);
	}
	return 0;
}
