/* replay: feed a traced instruction stream (QEMU -global ingenic-nna.log=nn.bin) through the
 * oracle and compare every read-back (NNDRD/NNRRD) with the recorded one. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
void *nna_oracle_new(void);
void ingenic_nna_op(void *opaque, unsigned op, uint32_t imm, uint8_t *vec);

int main(int argc, char **argv)
{
	FILE *f = fopen(argc > 1 ? argv[1] : "nn.bin", "rb");
	void *s = nna_oracle_new();
	struct { uint32_t op, imm; uint8_t vec[64]; } r;
	long n = 0, reads = 0, bad = 0;

	if (!f)
		return 2;
	while (fread(&r, sizeof(r), 1, f) == 1) {
		uint8_t v[64];

		if (r.op == 1 || r.op == 3) {		/* NNRRD / NNDRD: compare the output */
			memset(v, 0, 64);
			ingenic_nna_op(s, r.op, r.imm, v);
			reads++;
			if (memcmp(v, r.vec, 64)) {
				if (bad < 3)
					fprintf(stderr, "record %ld: op %u imm 0x%x differs\n", n, r.op, r.imm);
				bad++;
			}
		} else {
			memcpy(v, r.vec, 64);
			ingenic_nna_op(s, r.op, r.imm, v);
		}
		n++;
	}
	printf("%ld records replayed, %ld read-backs, %ld differ\n", n, reads, bad);
	return bad != 0;
}
