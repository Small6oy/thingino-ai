/*
 * nnappp — run the thingino-ai person/pet/package detector (2 heads, stride 32 + 16) on the
 * T41 NNA with venus, on pre-resized RGB frames, and print timings + boxes.
 *
 *   nnappp <model.bin> <loops> <conf> <frame.rgb>...     (each frame: 416x320x3 uint8 RGB)
 *   env NNAPPP_DUMP=dir: write <dir>/<n>.h32.f32 / .h16.f32 (NHWC float) for comparison
 *
 * Frames are already stretched to the network size (as raptor/aitest do: 16:9 -> 416x320), so
 * there is no letterbox and the camera result is directly comparable to the PC executor
 * (mgkrun) and the float model. Decode = thingino-ai train/qat/train.py decode().
 */
#include <cstddef>
#include <cstdint>
#include "venus.h"

#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace magik::venus;

static const int IW = 416, IH = 320, NC = 3;
static const char *NAMES[NC] = {"person", "pet", "package"};
static const float A32[6] = {81, 82, 135, 169, 344, 319}, A16[6] = {10, 14, 23, 27, 37, 58};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static float sigm(float x) { return 1.f / (1.f + expf(-x)); }

struct Det { float x1, y1, x2, y2, s; int c; };

/* p: NHWC [gh][gw][3 * (5 + NC)] */
static void decode(const float *p, int gh, int gw, int stride, const float *anc, float conf,
                   std::vector<Det> &out)
{
    const int oc = 5 + NC;
    for (int y = 0; y < gh; y++)
        for (int x = 0; x < gw; x++)
            for (int a = 0; a < 3; a++) {
                const float *q = p + ((long)y * gw + x) * oc * 3 + a * oc;
                float obj = sigm(q[4]);
                if (obj < conf)
                    continue;
                int bc = 0;
                float bs = 0;
                for (int c = 0; c < NC; c++) {
                    float v = sigm(q[5 + c]);
                    if (v > bs) bs = v, bc = c;
                }
                float s = obj * bs;
                if (s < conf)
                    continue;
                float cx = (sigm(q[0]) * 2 - 0.5f + x) * stride, cy = (sigm(q[1]) * 2 - 0.5f + y) * stride;
                float bw = sigm(q[2]) * 2, bh = sigm(q[3]) * 2;
                bw = bw * bw * anc[2 * a], bh = bh * bh * anc[2 * a + 1];
                out.push_back({cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, s, bc});
            }
}

static void nms(std::vector<Det> &in, std::vector<Det> &out, float thr)
{
    std::sort(in.begin(), in.end(), [](const Det &a, const Det &b) { return a.s > b.s; });
    std::vector<bool> gone(in.size(), false);
    for (size_t i = 0; i < in.size(); i++) {
        if (gone[i]) continue;
        out.push_back(in[i]);
        float ai = (in[i].x2 - in[i].x1) * (in[i].y2 - in[i].y1);
        for (size_t j = i + 1; j < in.size(); j++) {
            if (gone[j] || in[j].c != in[i].c) continue;
            float w = std::max(0.f, std::min(in[i].x2, in[j].x2) - std::max(in[i].x1, in[j].x1));
            float h = std::max(0.f, std::min(in[i].y2, in[j].y2) - std::max(in[i].y1, in[j].y1));
            float aj = (in[j].x2 - in[j].x1) * (in[j].y2 - in[j].y1);
            if (w * h / (ai + aj - w * h) > thr) gone[j] = true;
        }
    }
}

static long rss_kb(void)
{
    FILE *f = fopen("/proc/self/statm", "r");
    long a = 0, b = 0;
    if (f) { if (fscanf(f, "%ld %ld", &a, &b) != 2) b = 0; fclose(f); }
    return b * 4;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.bin loops conf frame.rgb...\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int loops = atoi(argv[2]);
    float conf = atof(argv[3]);
    const char *dump = getenv("NNAPPP_DUMP");

    /* the NN instructions only run on CPU0 */
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        perror("sched_setaffinity(CPU0)");

    long rss0 = rss_kb();
    double t0 = now_ms();
    if (venus_init() != 0) { fprintf(stderr, "venus_init failed\n"); return 1; }
    std::unique_ptr<BaseNet> net = net_create(TensorFormat::NHWC);
    if (!net || net->load_model(model) != 0) { fprintf(stderr, "load_model %s failed\n", model); return 1; }
    double t1 = now_ms();
    std::unique_ptr<Tensor> input = net->get_input(0);
    shape_t is = input->shape();
    printf("venus 0x%08x, load %.1f ms, venus mem %u B, RSS %ld -> %ld kB, input [%d %d %d %d]\n",
           venus_get_version_info(), t1 - t0, venus_get_used_mem_size(), rss0, rss_kb(), is[0], is[1], is[2], is[3]);
    input->reshape({1, IH, IW, 4});
    std::vector<std::string> on = net->get_output_names();
    for (auto &n : on) printf("output %s\n", n.c_str());

    std::vector<uint8_t> rgb((size_t)IW * IH * 3);
    for (int fi = 4; fi < argc; fi++) {
        FILE *fp = fopen(argv[fi], "rb");
        if (!fp || fread(rgb.data(), 1, rgb.size(), fp) != rgb.size()) {
            fprintf(stderr, "read %s failed\n", argv[fi]);
            if (fp) fclose(fp);
            continue;
        }
        fclose(fp);
        double tin = 0, trun = 0, tpost = 0, best = 1e9;
        std::vector<Det> kept;
        for (int it = 0; it < loops; it++) {
            double a = now_ms();
            uint8_t *d = input->mudata<uint8_t>();
            for (size_t i = 0, n = (size_t)IW * IH; i < n; i++) {
                d[4 * i] = rgb[3 * i]; d[4 * i + 1] = rgb[3 * i + 1]; d[4 * i + 2] = rgb[3 * i + 2]; d[4 * i + 3] = 0;
            }
            double b = now_ms();
            net->run();
            double c = now_ms();
            std::vector<Det> cand;
            for (auto &n : on) {
                std::unique_ptr<const Tensor> o = net->get_output_by_name(n);
                shape_t s = o->shape();
                const float *p = o->data<float>();
                int stride = IH / s[1];
                decode(p, s[1], s[2], stride, stride >= 32 ? A32 : A16, conf, cand);
                if (dump && it == 0) {
                    char path[512];
                    const char *base = strrchr(argv[fi], '/');
                    snprintf(path, sizeof(path), "%s/%s.%s.f32", dump, base ? base + 1 : argv[fi],
                             stride >= 32 ? "h32" : "h16");
                    FILE *df = fopen(path, "wb");
                    if (df) { fwrite(p, sizeof(float), (size_t)s[1] * s[2] * s[3], df); fclose(df); }
                }
            }
            kept.clear();
            nms(cand, kept, 0.45f);
            double e = now_ms();
            tin += b - a, trun += c - b, tpost += e - c;
            best = std::min(best, c - b);
        }
        const char *base = strrchr(argv[fi], '/');
        printf("== %s: input %.2f ms, run %.2f ms (best %.2f), post %.2f ms, %zu det\n", base ? base + 1 : argv[fi],
               tin / loops, trun / loops, best, tpost / loops, kept.size());
        for (auto &k : kept)
            printf("  %s %.2f [%d,%d,%d,%d]\n", NAMES[k.c], k.s, (int)k.x1, (int)k.y1, (int)k.x2, (int)k.y2);
    }
    printf("RSS %ld kB, venus mem %u B\n", rss_kb(), venus_get_used_mem_size());
    net.reset();
    venus_deinit();
    return 0;
}
