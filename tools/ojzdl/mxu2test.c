/* mxu2test: check every MXU2 instruction ojzdl uses against a C model of the
 * semantics documented in mxu2.h, on random vectors. Run under QEMU and on the
 * v3: the C models are what the kernels assume, so both must print PASS.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mxu2.h"

typedef union {
	int8_t b[16];
	int16_t h[8];
	int32_t w[4];
	uint8_t u[16];
} v128;

static uint32_t rng = 12345;
static uint32_t rnd(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	return rng;
}

static void fill(v128 *v)
{
	int i;
	for (i = 0; i < 16; i++)
		v->u[i] = (uint8_t)rnd();
}

static int fails;

static void check(const char *name, const v128 *got, const v128 *want, int it)
{
	if (memcmp(got, want, 16)) {
		int i;
		if (fails++ < 20) {
			printf("FAIL %s iter %d\n  got ", name, it);
			for (i = 0; i < 16; i++)
				printf("%02x", got->u[i]);
			printf("\n  want");
			for (i = 0; i < 16; i++)
				printf("%02x", want->u[i]);
			printf("\n");
		}
	}
}

/* run "vr1 = op(vr2, vr3[, vr4])" with vr1 preloaded from d (for accumulating ops) */
#define RUN3(INSN, d, s, t, r)                                                   \
	do {                                                                     \
		register v128 *pd __asm__("$8") = (d);                           \
		register const v128 *ps __asm__("$9") = (s);                     \
		register const v128 *pt __asm__("$10") = (t);                    \
		register const v128 *pr __asm__("$11") = (r);                    \
		__asm__ volatile(LU1Q(1, 0, 8) LU1Q(2, 0, 9) LU1Q(3, 0, 10) LU1Q(4, 0, 11) \
		                 INSN SU1Q(1, 0, 8)                              \
		                 : : "r"(pd), "r"(ps), "r"(pt), "r"(pr) : "memory"); \
	} while (0)

static int16_t sat16(int64_t v) { return (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v); }

int main(void)
{
	int it, i;
	static const uint8_t ilv[16] = { 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29 };

	for (it = 0; it < 2000; it++) {
		v128 s, t, r, d0, d, w;
		fill(&s); fill(&t); fill(&r); fill(&d0);

		/* DOTPS.h / DADDS.h */
		d = d0; RUN3(DOTPS(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++)
			w.h[i] = (int16_t)(uint16_t)(s.b[2 * i] * t.b[2 * i] + s.b[2 * i + 1] * t.b[2 * i + 1]);
		check("DOTPS.h", &d, &w, it);
		d = d0; RUN3(DADDS(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++)
			w.h[i] = (int16_t)(uint16_t)(d0.h[i] + s.b[2 * i] * t.b[2 * i] + s.b[2 * i + 1] * t.b[2 * i + 1]);
		check("DADDS.h", &d, &w, it);

		/* ADD.h / SUB.h (wrapping), SUB order */
		d = d0; RUN3(ADD(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = (int16_t)(uint16_t)(s.h[i] + t.h[i]);
		check("ADD.h", &d, &w, it);
		d = d0; RUN3(SUB(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = (int16_t)(uint16_t)(s.h[i] - t.h[i]);
		check("SUB.h", &d, &w, it);

		/* MULQ.h: sat16((s*t) >> 15) */
		d = d0; RUN3(MULQ_H(1, 2, 3), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = sat16(((int32_t)s.h[i] * t.h[i]) >> 15);
		check("MULQ.h", &d, &w, it);

		/* SRAR.h with n = t & 15 */
		d = d0; RUN3(SRAR(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) {
			int n = t.h[i] & 15;
			w.h[i] = (int16_t)(n ? (s.h[i] >> n) + ((s.h[i] >> (n - 1)) & 1) : s.h[i]);
		}
		check("SRAR.h", &d, &w, it);

		/* MAXS.h, MAXS.b */
		d = d0; RUN3(MAXS(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = s.h[i] > t.h[i] ? s.h[i] : t.h[i];
		check("MAXS.h", &d, &w, it);
		d = d0; RUN3(MAXS(1, 2, 3, DF_B), &d, &s, &t, &r);
		for (i = 0; i < 16; i++) w.b[i] = s.b[i] > t.b[i] ? s.b[i] : t.b[i];
		check("MAXS.b", &d, &w, it);
		d = d0; RUN3(MINS(1, 2, 3, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = s.h[i] < t.h[i] ? s.h[i] : t.h[i];
		check("MINS.h", &d, &w, it);

		/* SATS.h to 4/5/8 bits */
		d = d0; RUN3(SATS(1, 2, 5, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = (int16_t)(s.h[i] > 15 ? 15 : s.h[i] < -16 ? -16 : s.h[i]);
		check("SATS.h 5", &d, &w, it);
		d = d0; RUN3(SATS(1, 2, 4, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = (int16_t)(s.h[i] > 7 ? 7 : s.h[i] < -8 ? -8 : s.h[i]);
		check("SATS.h 4", &d, &w, it);

		/* SHUFV with a random selector, and with the narrowing selector */
		d = d0; RUN3(SHUFV(1, 2, 3, 4), &d, &s, &t, &r);
		for (i = 0; i < 16; i++) {
			uint8_t sel = r.u[i];
			int k = sel & 31;
			w.u[i] = sel & 0x80 ? 0 : (k & 1) ? s.u[k >> 1] : t.u[k >> 1];
		}
		check("SHUFV", &d, &w, it);
		memcpy(&r, ilv, 16);
		d = d0; RUN3(SHUFV(1, 2, 3, 4), &d, &s, &t, &r); /* t = lo halfwords, s = hi */
		for (i = 0; i < 8; i++) {
			w.b[i] = (int8_t)t.h[i];
			w.b[8 + i] = (int8_t)s.h[i];
		}
		check("SHUFV narrow", &d, &w, it);

		/* word forms used by the first-conv requant */
		d = d0; RUN3(DOTPS(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) w.w[i] = s.h[2 * i] * t.h[2 * i] + s.h[2 * i + 1] * t.h[2 * i + 1];
		check("DOTPS.w", &d, &w, it);
		d = d0; RUN3(ADD(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) w.w[i] = (int32_t)((uint32_t)s.w[i] + (uint32_t)t.w[i]);
		check("ADD.w", &d, &w, it);
		d = d0; RUN3(MULQ_W(1, 2, 3), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) {
			int64_t v = ((int64_t)s.w[i] * t.w[i]) >> 31;
			w.w[i] = (int32_t)(v > INT32_MAX ? INT32_MAX : v);
		}
		check("MULQ.w", &d, &w, it);
		d = d0; RUN3(SRAR(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) {
			int n = t.w[i] & 31;
			w.w[i] = n ? (s.w[i] >> n) + ((s.w[i] >> (n - 1)) & 1) : s.w[i];
		}
		check("SRAR.w", &d, &w, it);
		d = d0; RUN3(MAXS(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) w.w[i] = s.w[i] > t.w[i] ? s.w[i] : t.w[i];
		check("MAXS.w", &d, &w, it);
		d = d0; RUN3(MINS(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) w.w[i] = s.w[i] < t.w[i] ? s.w[i] : t.w[i];
		check("MINS.w", &d, &w, it);
		d = d0; RUN3(SUB(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++) w.w[i] = (int32_t)((uint32_t)s.w[i] - (uint32_t)t.w[i]);
		check("SUB.w", &d, &w, it);
		d = d0; RUN3(ORV(1, 2, 3), &d, &s, &t, &r);
		for (i = 0; i < 16; i++) w.u[i] = s.u[i] | t.u[i];
		check("ORV", &d, &w, it);
		/* DADDS.w (head widening) */
		d = d0; RUN3(DADDS(1, 2, 3, DF_W), &d, &s, &t, &r);
		for (i = 0; i < 4; i++)
			w.w[i] = (int32_t)((uint32_t)d0.w[i] + (uint32_t)(s.h[2 * i] * t.h[2 * i] + s.h[2 * i + 1] * t.h[2 * i + 1]));
		check("DADDS.w", &d, &w, it);
		/* LU1QX */
		{
			register v128 *pd __asm__("$8") = &d;
			register const v128 *ps __asm__("$9") = &s;
			register int ix __asm__("$10") = 0;
			__asm__ volatile(LU1QX(5, 9, 10) SU1Q(5, 0, 8) : : "r"(pd), "r"(ps), "r"(ix) : "memory");
			check("LU1QX", &d, &s, it);
		}

		d = d0; RUN3(XORIB(1, 2, 0x80), &d, &s, &t, &r);
		for (i = 0; i < 16; i++) w.u[i] = s.u[i] ^ 0x80;
		check("XORIB", &d, &w, it);

		d = d0; RUN3(SRAI(1, 2, 1, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = (int16_t)(s.h[i] >> 1);
		check("SRAI.h 1", &d, &w, it);
		d = d0; RUN3(SRARI(1, 2, 2, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = (int16_t)((s.h[i] >> 2) + ((s.h[i] >> 1) & 1));
		check("SRARI.h 2", &d, &w, it);

		/* REPI.h lane 5, LI.h -16 */
		d = d0; RUN3(REPI(1, 2, 5, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = s.h[5];
		check("REPI.h", &d, &w, it);
		d = d0; RUN3(LI(1, -16, DF_H), &d, &s, &t, &r);
		for (i = 0; i < 8; i++) w.h[i] = -16;
		check("LI.h", &d, &w, it);
	}
	/* unaligned load/store */
	{
		uint8_t buf[64], out[64];
		for (i = 0; i < 64; i++) { buf[i] = (uint8_t)rnd(); out[i] = 0; }
		{
			register uint8_t *a __asm__("$8") = buf + 3;
			register uint8_t *b __asm__("$9") = out + 5;
			__asm__ volatile(LU1Q(7, 17, 8) SU1Q(7, -3, 9) : : "r"(a), "r"(b) : "memory");
		}
		if (memcmp(out + 2, buf + 20, 16)) {
			printf("FAIL LU1Q/SU1Q unaligned offsets\n");
			fails++;
		}
	}
	printf("%s (%d failures)\n", fails ? "MXU2 SELFTEST FAIL" : "MXU2 SELFTEST PASS", fails);
	return fails != 0;
}
