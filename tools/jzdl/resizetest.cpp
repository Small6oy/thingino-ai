// resizetest: run libjzdl's jzdl::resize on one RGB image and dump the result,
// to pin down its interpolation (for an exact open re-implementation).
//
//   resizetest IN.rgb W H OUT.rgb OW OH
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include "net.h"
#include "utils.h"

int main(int argc, char **argv)
{
	if (argc < 7) {
		fprintf(stderr, "usage: %s IN.rgb W H OUT.rgb OW OH\n", argv[0]);
		return 2;
	}
	int w = atoi(argv[2]), h = atoi(argv[3]), ow = atoi(argv[5]), oh = atoi(argv[6]);
	std::vector<uint8_t> in((size_t)w * h * 3);
	FILE *f = fopen(argv[1], "rb");
	if (!f || fread(in.data(), 1, in.size(), f) != in.size())
		return 1;
	fclose(f);
	jzdl::Mat<uint8_t> src(w, h, 3, in.data());
	jzdl::Mat<uint8_t> dst(ow, oh, 3);
	jzdl::resize(src, dst);
	f = fopen(argv[4], "wb");
	if (!f || fwrite(dst.data, 1, (size_t)ow * oh * 3, f) != (size_t)ow * oh * 3)
		return 1;
	fclose(f);
	return 0;
}
