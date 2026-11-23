// jzdlrun: run a JZDL model on one raw input and dump raw outputs (step 2 reference).
//
//   jzdlrun MODEL.bin INPUT.rgb OUTPREFIX [blob ...]
//
// INPUT.rgb = the net's input size as uint8 RGB, HWC (w*h*3 bytes). It goes in
// exactly as aitest/raptor feed it: image_sub(128), then the bytes as int8.
// For each blob index given (default: the model's output, -999) it writes
// OUTPREFIX.<blob>.f32 = int32 w, h, c, then w*h*c floats as stored in Mat::data
// (no reordering), and prints the shape. Runs under qemu-mipsel -cpu XBurstR1
// as well as on the v3.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include "net.h"
#include "utils.h"

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: %s MODEL.bin INPUT.rgb OUTPREFIX [blob ...]\n", argv[0]);
		return 2;
	}
	jzdl::BaseNet *net = jzdl::net_create();
	if (!net || net->load_model(argv[1]) != 0) {
		fprintf(stderr, "cannot load %s\n", argv[1]);
		return 1;
	}
	std::vector<uint32_t> s = net->get_input_shape();
	int w = s[0], h = s[1], c = s[2];
	printf("input %dx%dx%d (w h c), model in blob %d out blob %d\n", w, h, c,
	       net->get_model_input_index(), net->get_model_output_index());
	std::vector<uint8_t> rgb((size_t)w * h * c);
	FILE *f = fopen(argv[2], "rb");
	if (!f || fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
		fprintf(stderr, "cannot read %zu bytes from %s\n", rgb.size(), argv[2]);
		return 1;
	}
	fclose(f);
	jzdl::Mat<uint8_t> img(w, h, c, rgb.data());
	jzdl::image_sub(img, 128);
	jzdl::Mat<int8_t> in(w, h, c, (int8_t *)img.data);
	if (net->input(in) != 0) {
		fprintf(stderr, "input failed\n");
		return 1;
	}
	std::vector<int> blobs;
	for (int i = 4; i < argc; i++)
		blobs.push_back(atoi(argv[i]));
	if (blobs.empty())
		blobs.push_back(-999);
	for (size_t k = 0; k < blobs.size(); k++) {
		jzdl::Mat<float> out;
		if (net->run(out, blobs[k]) != 0) {
			fprintf(stderr, "run(blob %d) failed\n", blobs[k]);
			return 1;
		}
		size_t n = out.total();
		char name[512];
		snprintf(name, sizeof(name), "%s.%d.f32", argv[3], blobs[k]);
		FILE *o = fopen(name, "wb");
		int32_t hdr[3] = { out.w, out.h, out.c };
		fwrite(hdr, 4, 3, o);
		fwrite(out.data, sizeof(float), n, o);
		fclose(o);
		float mn = n ? out.data[0] : 0, mx = mn;
		for (size_t i = 0; i < n; i++) {
			if (out.data[i] < mn) mn = out.data[i];
			if (out.data[i] > mx) mx = out.data[i];
		}
		printf("blob %d: w %d h %d c %d dims %d step %zu total %zu min %g max %g -> %s\n",
		       blobs[k], out.w, out.h, out.c, out.dims, out.step, n, mn, mx, name);
	}
	jzdl::net_destory(net);
	return 0;
}
