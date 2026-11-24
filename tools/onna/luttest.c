/* luttest: the MXU3 256-entry byte table (LUT256) against a plain lookup, every byte value,
 * random tables. Runs under QEMU -cpu XBurst2-MXU3.1 and on the v4 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "mxu3.h"

int main(void)
{
	static uint8_t tab[256] __attribute__((aligned(64))), x[256] __attribute__((aligned(64))),
		y[256] __attribute__((aligned(64)));
	long bad = 0;

	srand(1);
	for (int r = 0; r < 200; r++) {
		for (int i = 0; i < 256; i++)
			tab[i] = (uint8_t)rand(), x[i] = (uint8_t)(r ? rand() : i);
		VLOAD(20, tab);
		VLOAD(21, tab + 64);
		VLOAD(22, tab + 128);
		VLOAD(23, tab + 192);
		for (int v = 0; v < 256; v += 64) {
			VLOAD(1, x + v);
			__asm__ volatile(LUT256(3, 1, 20, 4) ::: "memory");
			VSTORE(3, y + v);
		}
		for (int i = 0; i < 256; i++)
			bad += y[i] != tab[x[i]];
	}
	printf("luttest: %ld of %d bytes differ: %s\n", bad, 200 * 256, bad ? "FAIL" : "PASS");
	return bad != 0;
}
