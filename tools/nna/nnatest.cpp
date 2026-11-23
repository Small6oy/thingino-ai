/*
 * nnatest — run a Magik/venus YOLOv5 model on the T41 NNA against an NV12 frame.
 *
 *   nnatest <model.bin> <frame.nv12> <w> <h> [in_w in_h] [loops] [classes] [conf]
 *
 * Letterboxes the frame on the AIP (NV12 -> RGBA), runs the net on the NNA,
 * decodes YOLOv5 boxes + per-class NMS on the CPU, and prints timings and
 * boxes in frame coordinates. Based on the vendor sample
 * magik-toolkit/Models/post/yolov5s/venus_sample_yolov5s/inference_nv12.cpp.
 */
#include <cstddef> /* vendor headers use size_t without including it (GCC 15) */
#include <cstdint>
#include "venus.h"

#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <memory>
#include <string>
#include <vector>

using namespace magik::venus;

static const char *COCO[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* NNATEST_DUMP=dir: print tensor stats and write the resized input (RGBA) and outputs (float) */
static void stats(const char *tag, const float *f, size_t n)
{
    double sum = 0;
    float mn = 1e30f, mx = -1e30f;
    size_t nz = 0, nan = 0;
    for (size_t i = 0; i < n; i++) {
        float v = f[i];
        if (v != v) { nan++; continue; }
        if (v != 0) nz++;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    printf("dump %s: n=%zu min=%.4f max=%.4f mean=%.4f nonzero=%zu nan=%zu\n", tag, n, mn, mx,
           n ? sum / n : 0.0, nz, nan);
}

static void dump_file(const char *dir, const char *name, const void *p, size_t n)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(p, 1, n, f);
        fclose(f);
    }
}

static long rss_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[128];
    long kb = -1;
    if (!f)
        return -1;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "VmRSS: %ld kB", &kb) == 1)
            break;
    fclose(f);
    return kb;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.bin frame.nv12 w h [in_w in_h] [loops] [classes] [conf]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *frame = argv[2];
    int w = atoi(argv[3]), h = atoi(argv[4]);
    int in_w = argc > 6 ? atoi(argv[5]) : 640;
    int in_h = argc > 6 ? atoi(argv[6]) : 384;
    int loops = argc > 7 ? atoi(argv[7]) : 10;
    int classes = argc > 8 ? atoi(argv[8]) : 80;
    float conf = argc > 9 ? atof(argv[9]) : 0.3f;
    float nms_thr = 0.6f;

    /* Only core 0 of the T41 reaches the NNA: the MXU3 NN instructions are Reserved
     * Instruction on CPU1 (tools/nnprobe.c), so pin before venus runs any of them,
     * as Ingenic's inference_nv12.cpp does. */
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        perror("sched_setaffinity(CPU0)");

    /* frame */
    size_t nv12_size = (size_t)w * h * 3 / 2;
    std::vector<uint8_t> nv12(nv12_size);
    FILE *fp = fopen(frame, "rb");
    if (!fp || fread(nv12.data(), 1, nv12_size, fp) != nv12_size) {
        fprintf(stderr, "read %s failed (need %zu bytes)\n", frame, nv12_size);
        return 1;
    }
    fclose(fp);

    long rss0 = rss_kb();
    double t0 = now_ms();
    if (venus_init() != 0) {
        fprintf(stderr, "venus_init failed\n");
        return 1;
    }
    printf("venus version info: 0x%08x\n", venus_get_version_info());
    double t1 = now_ms();

    std::unique_ptr<BaseNet> net = net_create(TensorFormat::NHWC);
    if (!net || net->load_model(model) != 0) {
        fprintf(stderr, "load_model %s failed\n", model);
        return 1;
    }
    double t2 = now_ms();
    printf("init %.1f ms, load %.1f ms, venus mem %u B, RSS %ld -> %ld kB\n", t1 - t0, t2 - t1,
           venus_get_used_mem_size(), rss0, rss_kb());

    std::unique_ptr<Tensor> input = net->get_input(0);
    shape_t ishape = input->shape();
    printf("model input %d x %d x %d -> using %d x %d\n", ishape[1], ishape[2], ishape[3], in_h,
           in_w);
    input->reshape({1, in_h, in_w, 4});

    /* NV12 source lives in NNA-visible memory (a venus tensor) */
    Tensor src({1, h, w, 1}, TensorFormat::NV12);
    memcopy((void *)src.mudata<uint8_t>(), (void *)nv12.data(), nv12_size);

    float scale = fminf((float)in_w / w, (float)in_h / h);
    int vw = (int)(scale * w), vh = (int)(scale * h);
    vw += vw & 1;
    vh += vh & 1;
    int pad_left = (int)roundf((in_w - vw) / 2.0f - 0.1f);
    int pad_top = (int)roundf((in_h - vh) / 2.0f - 0.1f);

    BsCommonParam param;
    param.pad_val = 0;
    param.pad_type = BsPadType::SYMMETRY;
    param.input_height = h;
    param.input_width = w;
    param.input_line_stride = w;
    param.in_layout = ChannelLayout::NV12;
    param.out_layout = ChannelLayout::RGBA;

    std::vector<float> strides = {8.0, 16.0, 32.0};
    std::vector<float> anchor = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    std::vector<ObjBbox_t> boxes;

    double t_rs = 0, t_run = 0, t_post = 0, best_run = 1e9;
    for (int it = 0; it < loops; it++) {
        double a = now_ms();
        common_resize((const void *)src.mudata<uint8_t>(), *input, AddressLocate::NMEM_VIRTUAL, &param);
        double b = now_ms();
        net->run();
        double c = now_ms();

        const char *dump = getenv("NNATEST_DUMP");
        if (dump && it == 0) {
            const uint8_t *in = input->data<uint8_t>();
            size_t n = (size_t)in_h * in_w * 4;
            size_t nz = 0;
            unsigned mx = 0;
            double sum = 0;
            for (size_t i = 0; i < n; i++) {
                nz += in[i] != 0;
                mx = in[i] > mx ? in[i] : mx;
                sum += in[i];
            }
            printf("dump input %dx%d RGBA: nonzero=%zu/%zu max=%u mean=%.2f\n", in_w, in_h, nz, n, mx,
                   sum / n);
            dump_file(dump, "input_rgba.raw", in, n);
        }

        std::vector<Tensor> feats;
        for (int i = 0; i < 3; i++) {
            std::unique_ptr<const Tensor> o = net->get_output(i);
            shape_t s = o->shape();
            Tensor t(s);
            size_t n = (size_t)s[0] * s[1] * s[2] * s[3];
            memcopy((void *)t.mudata<float>(), (void *)o->data<float>(), n * sizeof(float));
            if (dump && it == 0) {
                char tag[64], name[32];
                snprintf(tag, sizeof(tag), "out%d [%d %d %d %d]", i, s[0], s[1], s[2], s[3]);
                stats(tag, t.data<float>(), n);
                snprintf(name, sizeof(name), "out%d.f32", i);
                dump_file(dump, name, t.data<float>(), n * sizeof(float));
            }
            feats.push_back(t);
        }
        std::vector<ObjBbox_t> cand;
        generate_box(feats, strides, anchor, cand, in_w, in_h, classes, 3, conf, DetectorType::YOLOV5);
        boxes.clear();
        for (int cls = 0; cls < classes; cls++) {
            std::vector<ObjBbox_t> buf;
            for (auto &bx : cand)
                if (bx.class_id == cls)
                    buf.push_back(bx);
            if (!buf.empty())
                nms(buf, boxes, nms_thr, NmsType::HARD_NMS);
        }
        double d = now_ms();
        t_rs += b - a;
        t_run += c - b;
        t_post += d - c;
        if (c - b < best_run)
            best_run = c - b;
    }
    printf("per frame (avg of %d): resize %.2f ms, run %.2f ms (best %.2f), post %.2f ms\n", loops,
           t_rs / loops, t_run / loops, best_run, t_post / loops);
    printf("RSS after run %ld kB, venus mem %u B\n", rss_kb(), venus_get_used_mem_size());

    for (auto &bx : boxes) {
        int x0 = (int)((bx.box.x0 - pad_left) / scale), y0 = (int)((bx.box.y0 - pad_top) / scale);
        int x1 = (int)((bx.box.x1 - pad_left) / scale), y1 = (int)((bx.box.y1 - pad_top) / scale);
        const char *name = (bx.class_id >= 0 && bx.class_id < 80 && classes == 80) ? COCO[bx.class_id] : "?";
        printf("box %-12s cls=%2d score=%.2f  [%d,%d %d,%d]\n", name, bx.class_id, bx.score, x0, y0,
               x1, y1);
    }

    net.reset();
    venus_deinit();
    return 0;
}
