/* mxu3test: self-test of the MXU3 data path onna's backend relies on (LAO/SAO round trip,
 * MFCPUW/MTCPUW). Runs under user-mode QEMU (-cpu XBurst2) and on the camera; the NN
 * instructions themselves need the NNA (system QEMU or silicon). Prints PASS or FAIL. */
#include <stdio.h>
#include <string.h>
#include "mxu3.h"

int main(void)
{
	unsigned char a[64] __attribute__((aligned(64))), b[64] __attribute__((aligned(64)));
	unsigned w = 0;
	int fail = 0;

	for (int i = 0; i < 64; i++)
		a[i] = (unsigned char)(i * 7 + 3);
	memset(b, 0, sizeof(b));
	VPR0_LOAD(a);
	VPR0_STORE(b);
	if (memcmp(a, b, 64)) {
		printf("LAO/SAO round trip: FAIL\n");
		fail = 1;
	}
	VWR0_SET(0x12345678u);
	VWR0_GET(w);
	if (w != 0x12345678u) {
		printf("MFCPUW/MTCPUW: FAIL (0x%08x)\n", w);
		fail = 1;
	}
	printf("%s\n", fail ? "FAIL" : "PASS");
	return fail;
}
