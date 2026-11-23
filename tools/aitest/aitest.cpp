/*
 * aitest -- G100-style detection benchmark for the Wyze v4 (T41).
 *
 * Copies the Aqara G100's design (nna/aqara-g100-ai.md): a separate process asks the
 * video daemon for ONE small NV12 frame at a fixed interval (default 1 s), reads it
 * from a tmpfs file and runs a detector on it. rvd and raptor.conf are not changed:
 * frames come from rvd's existing control command
 *   {"cmd":"save","format":"raw","file":"/tmp/360.yuv","channel":1}
 * (the same as `raptorctl rvd save raw /tmp/360.yuv 1`).
 *
 * Backends (same loop, same frames, so the numbers compare):
 *   jzdl   Ingenic JZDL on the CPU (libjzdl.m.so), Ingenic's yolov5 JZDL model
 *   ncnn   ncnn on the CPU, plain C++ (no SIMD), our yolov5n (same graph as the NNA's)
 *   venus  (later) Magik venus on the NNA, our converted yolov5n
 *
 * Every report period it prints per-phase times (grab, pre, infer, post), its own CPU %,
 * rvd's CPU %, the system's busy %, RSS and the detections.
 *
 * usage: aitest [-b jzdl|ncnn] [-m model] [-x net_w -y net_h (ncnn)] [-i interval_ms] [-d seconds] [-r report_s]
 *               [-c classes] [-t conf%] [-C channel] [-f frame_file] [-F file_once]
 *               [-p cpu] [-q]
 */
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef AITEST_JZDL
#include "net.h"
#include "utils.h"
#endif
#ifdef AITEST_NCNN
#include "ncnn/net.h"
#endif

extern "C" int rss_ctrl_send_command(const char *sock_path, const char *cmd_json, char *resp_buf,
				     int resp_buf_size, int timeout_ms);

static volatile sig_atomic_t g_stop;
static void on_sig(int) { g_stop = 1; }

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ---------------- detections ---------------- */

struct Det {
	float x1, y1, x2, y2, score;
	int cls;
};

/* Ingenic yolov5 JZDL model: person, bicycle, car, motorcycle (raptor's default). */
static const char *k_names4[] = {"person", "bicycle", "car", "motorcycle"};
/* COCO: the first four match the JZDL model; others print as their index */
static const char *cls_name(int c)
{
	static char b[8];
	if (c >= 0 && c < 4)
		return k_names4[c];
	snprintf(b, sizeof(b), "c%d", c);
	return b;
}

/* ---------------- backend interface ---------------- */

struct Backend {
	virtual ~Backend() {}
	virtual bool load(const char *model) = 0;
	virtual int in_w() const = 0;
	virtual int in_h() const = 0;
	/* nv12 at fw x fh -> detections in frame coordinates; fills phase times (ms) */
	virtual bool detect(const uint8_t *nv12, int fw, int fh, std::vector<Det> &out,
			    double &t_pre, double &t_inf, double &t_post) = 0;
};

/* ---- YOLOv5 decode + NMS (same maths as raptor-hal hal_ivs_jzdl.cpp) ---- */

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

#ifdef AITEST_JZDL
static void yolo_decode(const float *p, std::vector<Det> &boxes, int iw, int ih, float conf,
			int ncls)
{
	static const float anchors[] = {10, 13, 16, 30, 33, 23, 30, 61, 62,
					45, 59, 119, 116, 90, 156, 198, 373, 326};
	static const float strides[] = {8.f, 16.f, 32.f};
	const int nb = 3, oc = 5 + ncls;
	for (int s = 0; s < 3; s++) {
		int gh = ih / (int)strides[s], gw = iw / (int)strides[s];
		for (int h = 0; h < gh; h++)
			for (int w = 0; w < gw; w++)
				for (int n = 0; n < nb; n++) {
					const float *q = p + h * (gw * nb * oc) + w * (nb * oc) + n * oc;
					float obj = sigm(q[4]);
					if (obj < conf)
						continue; /* cheap early out; prob = obj * cls <= obj */
					int bc = 0;
					float bs = 0;
					for (int c = 0; c < ncls; c++) {
						float cs = sigm(q[5 + c]);
						if (cs > bs) {
							bs = cs;
							bc = c;
						}
					}
					float prob = obj * bs;
					if (prob < conf)
						continue;
					float aw = anchors[s * nb * 2 + n * 2] / iw;
					float ah = anchors[s * nb * 2 + n * 2 + 1] / ih;
					float bw = sigm(q[2]) * 2, bh = sigm(q[3]) * 2;
					float box_w = bw * bw * aw, box_h = bh * bh * ah;
					float cx = (sigm(q[0]) * 2 - 0.5f + w) / gw;
					float cy = (sigm(q[1]) * 2 - 0.5f + h) / gh;
					Det d = {(cx - box_w / 2) * iw, (cy - box_h / 2) * ih,
						 (cx + box_w / 2) * iw, (cy + box_h / 2) * ih, prob, bc};
					boxes.push_back(d);
				}
		p += gh * gw * nb * oc;
	}
}

/* Ingenic's JZDL persondet (magik_model_persondet.bin, 416x320): K YOLO heads, each
 * HWC [h][w][3 anchors x (5 + classes)], back to back, then a trailer of K x (h, w, c)
 * and K, e.g. 10 13 18  20 26 18  2. Same cell maths as YOLOv5; the anchors aren't
 * published - the yolov3-tiny ones fit the boxes on test images (stride 32 and 16). */
static bool headed_decode(const float *p, long n, std::vector<Det> &boxes, int iw, int ih,
			  float conf)
{
	if (n < 7)
		return false;
	int k = (int)p[n - 1];
	if (k < 1 || k > 4 || n < 1 + 3 * k)
		return false;
	const float *tr = p + n - 1 - 3 * k;
	long sum = 0;
	for (int i = 0; i < k; i++)
		sum += (long)tr[3 * i] * tr[3 * i + 1] * tr[3 * i + 2];
	if (sum + 1 + 3 * k != n)
		return false;
	static const float a32[] = {81, 82, 135, 169, 344, 319}, a16[] = {10, 14, 23, 27, 37, 58};
	for (int i = 0; i < k; i++) {
		int gh = (int)tr[3 * i], gw = (int)tr[3 * i + 1], oc = (int)tr[3 * i + 2] / 3;
		int ncls = oc - 5;
		const float *an = (iw / gw >= 32) ? a32 : a16;
		for (int y = 0; y < gh; y++)
			for (int x = 0; x < gw; x++)
				for (int a = 0; a < 3; a++) {
					const float *q = p + ((long)y * gw + x) * oc * 3 + a * oc;
					float obj = sigm(q[4]);
					if (obj < conf)
						continue;
					int bc = 0;
					float bs = 0;
					for (int c = 0; c < ncls; c++)
						if (sigm(q[5 + c]) > bs)
							bs = sigm(q[5 + c]), bc = c;
					float prob = obj * bs;
					if (prob < conf)
						continue;
					float bw = sigm(q[2]) * 2, bh = sigm(q[3]) * 2;
					float box_w = bw * bw * an[2 * a], box_h = bh * bh * an[2 * a + 1];
					float cx = (sigm(q[0]) * 2 - 0.5f + x) * iw / gw;
					float cy = (sigm(q[1]) * 2 - 0.5f + y) * ih / gh;
					Det d = {cx - box_w / 2, cy - box_h / 2, cx + box_w / 2,
						 cy + box_h / 2, prob, bc};
					boxes.push_back(d);
				}
		p += (long)gh * gw * oc * 3;
	}
	return true;
}

#endif

static void nms(std::vector<Det> &in, std::vector<Det> &out, float thr)
{
	std::sort(in.begin(), in.end(), [](const Det &a, const Det &b) { return a.score > b.score; });
	std::vector<bool> gone(in.size(), false);
	for (size_t i = 0; i < in.size(); i++) {
		if (gone[i])
			continue;
		out.push_back(in[i]);
		float ai = (in[i].x2 - in[i].x1) * (in[i].y2 - in[i].y1);
		for (size_t j = i + 1; j < in.size(); j++) {
			if (gone[j] || in[j].cls != in[i].cls)
				continue;
			float iw = std::max(0.f, std::min(in[i].x2, in[j].x2) - std::max(in[i].x1, in[j].x1));
			float ih = std::max(0.f, std::min(in[i].y2, in[j].y2) - std::max(in[i].y1, in[j].y1));
			float inter = iw * ih;
			float aj = (in[j].x2 - in[j].x1) * (in[j].y2 - in[j].y1);
			if (inter / (ai + aj - inter) > thr)
				gone[j] = true;
		}
	}
}

/* NV12 -> RGB, scaled straight to the network input (nearest neighbour). raptor converts
 * the full frame and then resizes; doing both at once saves most of the per-pixel work.
 * Both are timed as "pre". */
static void nv12_to_rgb_scaled(const uint8_t *nv12, int fw, int fh, uint8_t *rgb, int ow, int oh)
{
	const uint8_t *yp = nv12, *uvp = nv12 + fw * fh;
	for (int r = 0; r < oh; r++) {
		int sy = r * fh / oh;
		for (int c = 0; c < ow; c++) {
			int sx = c * fw / ow;
			int y = yp[sy * fw + sx];
			int u = uvp[(sy / 2) * fw + (sx & ~1)] - 128;
			int v = uvp[(sy / 2) * fw + (sx | 1)] - 128;
			int R = y + ((359 * v) >> 8), G = y - ((88 * u + 183 * v) >> 8), B = y + ((454 * u) >> 8);
			uint8_t *o = rgb + (r * ow + c) * 3;
			o[0] = R < 0 ? 0 : R > 255 ? 255 : R;
			o[1] = G < 0 ? 0 : G > 255 ? 255 : G;
			o[2] = B < 0 ? 0 : B > 255 ? 255 : B;
		}
	}
}

#ifdef AITEST_JZDL
/* ---- JZDL (CPU) ---- */

struct JzdlBackend : Backend {
	jzdl::BaseNet *net = nullptr;
	int w = 0, h = 0, c = 0, ncls;
	bool checked = false, bad = false, headed = false, bgr = getenv("AITEST_BGR") != nullptr;
	float conf, nmsth;
	std::vector<uint8_t> rgb;
	std::vector<Det> cand, kept;
	JzdlBackend(int ncls_, float conf_, float nms_) : ncls(ncls_), conf(conf_), nmsth(nms_) {}
	~JzdlBackend() { if (net) jzdl::net_destory(net); }
	bool load(const char *model) override
	{
		net = jzdl::net_create();
		if (!net || net->load_model(model) != 0) {
			fprintf(stderr, "jzdl: cannot load %s\n", model);
			return false;
		}
		std::vector<uint32_t> s = net->get_input_shape();
		if (s.size() < 3)
			return false;
		w = s[0], h = s[1], c = s[2];
		rgb.resize((size_t)w * h * 3);
		printf("jzdl: model %s input %dx%dx%d, %d classes\n", model, w, h, c, ncls);
		return true;
	}
	int in_w() const override { return w; }
	int in_h() const override { return h; }
	bool detect(const uint8_t *nv12, int fw, int fh, std::vector<Det> &out, double &tp,
		    double &ti, double &tq) override
	{
		double t0 = now_ms();
		nv12_to_rgb_scaled(nv12, fw, fh, rgb.data(), w, h);
		if (bgr)
			for (size_t i = 0; i < rgb.size(); i += 3)
				std::swap(rgb[i], rgb[i + 2]);
		jzdl::Mat<uint8_t> img(w, h, 3, rgb.data());
		jzdl::image_sub(img, 128);
		jzdl::Mat<int8_t> in(w, h, c, (int8_t *)img.data);
		double t1 = now_ms();
		jzdl::Mat<float> o;
		net->input(in);
		net->run(o);
		double t2 = now_ms();
		cand.clear();
		kept.clear();
		out.clear();
		if (o.data && !checked) {
			/* the output size tells the class count (YOLOv5 head: 3 anchors x
			 * (5 + classes) per cell of the stride 8/16/32 grids) */
			long n = (long)o.w * o.h * o.c, cells = 0;
			for (int st = 8; st <= 32; st *= 2)
				cells += (long)(w / st) * (h / st);
			printf("jzdl: output %dx%dx%d = %ld floats, %ld grid cells\n", o.w, o.h, o.c,
			       n, cells);
			if (n > 7 && headed_decode(o.data, n, cand, w, h, 2.f)) {
				headed = true;
				printf("jzdl: output has %d heads + shape trailer (persondet "
				       "layout)\n", (int)o.data[n - 1]);
			} else if (n % (cells * 3) == 0 && n / (cells * 3) > 5) {
				if (n / (cells * 3) - 5 != ncls)
					printf("jzdl: output says %ld classes (not %d), using that\n",
					       n / (cells * 3) - 5, ncls);
				ncls = (int)(n / (cells * 3) - 5);
			} else {
				fprintf(stderr, "jzdl: output is not a YOLOv5 head, no decode; raw "
						"floats written to /tmp/ai/out.f32\n");
				FILE *df = fopen("/tmp/ai/out.f32", "wb");
				if (df) {
					fwrite(o.data, sizeof(float), n, df);
					fclose(df);
				}
				bad = true;
			}
			checked = true;
		}
		if (o.data && !bad) {
			if (headed)
				headed_decode(o.data, (long)o.w * o.h * o.c, cand, w, h, conf);
			else
				yolo_decode(o.data, cand, w, h, conf, ncls);
			nms(cand, kept, nmsth);
		}
		float sx = (float)fw / w, sy = (float)fh / h;
		for (auto &d : kept)
			out.push_back({d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, d.score, d.cls});
		double t3 = now_ms();
		tp = t1 - t0, ti = t2 - t1, tq = t3 - t2;
		return true;
	}
};

#endif /* AITEST_JZDL */

#ifdef AITEST_NCNN
/* ---- ncnn (CPU, plain C++: no MSA/MMI/OpenMP) ----
 * Model: our yolov5n ONNX cut at the three Detect convs (the same graph Magik converts
 * for the NNA), so outputs are raw NCHW [3*(5+80), gh, gw]. Input size is a runtime
 * choice (-W/-H); the conv-only graph accepts any multiple of 32. */
struct NcnnBackend : Backend {
	ncnn::Net net;
	int w, h, ncls;
	float conf, nmsth;
	bool lowmem;
	std::vector<uint8_t> rgb;
	std::vector<Det> cand, kept;
	NcnnBackend(int w_, int h_, int ncls_, float conf_, float nms_, bool lowmem_)
		: w(w_), h(h_), ncls(ncls_), conf(conf_), nmsth(nms_), lowmem(lowmem_) {}
	bool load(const char *model) override
	{
		/* model = path without extension: <model>.param + <model>.bin */
		char p[256];
		net.opt.num_threads = 1;
		net.opt.use_packing_layout = true;
		net.opt.lightmode = true; /* free intermediate blobs as soon as they're consumed */
		/* -L: no winograd/sgemm weight pre-transforms (they multiply the weight RAM;
		 * 42 MB RSS at 320x192 on the v3 with them on) */
		net.opt.use_winograd_convolution = !lowmem;
		net.opt.use_sgemm_convolution = !lowmem;
		snprintf(p, sizeof(p), "%s.param", model);
		if (net.load_param(p) != 0)
			return false;
		snprintf(p, sizeof(p), "%s.bin", model);
		if (net.load_model(p) != 0)
			return false;
		rgb.resize((size_t)w * h * 3);
		printf("ncnn: model %s input %dx%d, %d classes, 1 thread%s\n", model, w, h, ncls,
		       lowmem ? ", low-memory (no winograd/sgemm)" : "");
		return true;
	}
	int in_w() const override { return w; }
	int in_h() const override { return h; }
	void decode(const ncnn::Mat &m, int stride_i, std::vector<Det> &boxes)
	{
		static const float anchors[] = {10, 13, 16, 30, 33, 23, 30, 61, 62,
						45, 59, 119, 116, 90, 156, 198, 373, 326};
		const int oc = 5 + ncls, gh = m.h, gw = m.w;
		const float stride = (float)(8 << stride_i);
		for (int a = 0; a < 3; a++) {
			const float aw = anchors[stride_i * 6 + a * 2], ah = anchors[stride_i * 6 + a * 2 + 1];
			for (int y = 0; y < gh; y++)
				for (int x = 0; x < gw; x++) {
					int off = y * gw + x;
					float obj = sigm(m.channel(a * oc + 4)[off]);
					if (obj < conf)
						continue;
					int bc = 0;
					float bs = 0;
					for (int c = 0; c < ncls; c++) {
						float v = m.channel(a * oc + 5 + c)[off];
						if (v > bs || c == 0) {
							bs = v;
							bc = c;
						}
					}
					float prob = obj * sigm(bs);
					if (prob < conf)
						continue;
					float cx = (sigm(m.channel(a * oc + 0)[off]) * 2 - 0.5f + x) * stride;
					float cy = (sigm(m.channel(a * oc + 1)[off]) * 2 - 0.5f + y) * stride;
					float bw = sigm(m.channel(a * oc + 2)[off]) * 2;
					float bh = sigm(m.channel(a * oc + 3)[off]) * 2;
					bw = bw * bw * aw, bh = bh * bh * ah;
					Det d = {cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, prob, bc};
					boxes.push_back(d);
				}
		}
	}
	bool detect(const uint8_t *nv12, int fw, int fh, std::vector<Det> &out, double &tp,
		    double &ti, double &tq) override
	{
		static const char *outs[3] = {"/model.24/m.0/Conv_output_0", "/model.24/m.1/Conv_output_0",
					      "/model.24/m.2/Conv_output_0"};
		static const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
		double t0 = now_ms();
		nv12_to_rgb_scaled(nv12, fw, fh, rgb.data(), w, h);
		ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data(), ncnn::Mat::PIXEL_RGB, w, h);
		in.substract_mean_normalize(0, norm);
		double t1 = now_ms();
		ncnn::Extractor ex = net.create_extractor();
		ex.input("images", in);
		ncnn::Mat o[3];
		for (int i = 0; i < 3; i++)
			ex.extract(outs[i], o[i]);
		double t2 = now_ms();
		cand.clear();
		kept.clear();
		out.clear();
		for (int i = 0; i < 3; i++)
			decode(o[i], i, cand);
		nms(cand, kept, nmsth);
		float sx = (float)fw / w, sy = (float)fh / h;
		for (auto &d : kept)
			out.push_back({d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, d.score, d.cls});
		double t3 = now_ms();
		tp = t1 - t0, ti = t2 - t1, tq = t3 - t2;
		return true;
	}
};
#endif /* AITEST_NCNN */

/* ---------------- /proc helpers ---------------- */

static long proc_ticks(pid_t pid)
{
	char p[64], b[512];
	snprintf(p, sizeof(p), "/proc/%d/stat", (int)pid);
	int fd = open(p, O_RDONLY);
	if (fd < 0)
		return -1;
	ssize_t n = read(fd, b, sizeof(b) - 1);
	close(fd);
	if (n <= 0)
		return -1;
	b[n] = 0;
	char *s = strrchr(b, ')');
	if (!s)
		return -1;
	long ut = 0, st = 0;
	/* fields after ")": state(3) ... utime(14) stime(15) */
	if (sscanf(s + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %ld %ld", &ut, &st) != 2)
		return -1;
	return ut + st;
}

static void sys_ticks(long long &busy, long long &total)
{
	FILE *f = fopen("/proc/stat", "r");
	long long v[8] = {0};
	if (f) {
		if (fscanf(f, "cpu %lld %lld %lld %lld %lld %lld %lld %lld", &v[0], &v[1], &v[2], &v[3],
			   &v[4], &v[5], &v[6], &v[7]) < 4) {}
		fclose(f);
	}
	total = 0;
	for (int i = 0; i < 8; i++)
		total += v[i];
	busy = total - v[3] - v[4];
}

static long rss_kb(void)
{
	FILE *f = fopen("/proc/self/status", "r");
	char l[128];
	long kb = -1;
	while (f && fgets(l, sizeof(l), f))
		if (!strncmp(l, "VmRSS:", 6))
			kb = atol(l + 6);
	if (f)
		fclose(f);
	return kb;
}

/* MemAvailable, or MemFree + Buffers + Cached on kernels before 3.14 (the v3's 3.10) */
static long mem_avail_kb(void)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char l[128];
	long avail = -1, sum = 0;
	while (f && fgets(l, sizeof(l), f)) {
		if (!strncmp(l, "MemAvailable:", 13))
			avail = atol(l + 13);
		else if (!strncmp(l, "MemFree:", 8))
			sum += atol(l + 8);
		else if (!strncmp(l, "Buffers:", 8))
			sum += atol(l + 8);
		else if (!strncmp(l, "Cached:", 7))
			sum += atol(l + 7);
	}
	if (f)
		fclose(f);
	return avail >= 0 ? avail : sum;
}

static pid_t pid_of(const char *name)
{
	char p[64], b[64];
	snprintf(p, sizeof(p), "/var/run/%s.pid", name);
	FILE *f = fopen(p, "r");
	if (f) {
		int pid = 0;
		if (fscanf(f, "%d", &pid) == 1 && pid > 0) {
			fclose(f);
			return pid;
		}
		fclose(f);
	}
	/* fall back to scanning /proc/<pid>/comm */
	for (int pid = 2; pid < 32768; pid++) {
		snprintf(p, sizeof(p), "/proc/%d/comm", pid);
		FILE *c = fopen(p, "r");
		if (!c)
			continue;
		bool m = fgets(b, sizeof(b), c) && !strncmp(b, name, strlen(name)) && b[strlen(name)] == '\n';
		fclose(c);
		if (m)
			return pid;
	}
	return -1;
}

/* ---------------- frame source ---------------- */

static bool read_frame(const char *file, std::vector<uint8_t> &buf)
{
	int fd = open(file, O_RDONLY);
	if (fd < 0)
		return false;
	struct stat st;
	if (fstat(fd, &st) != 0 || st.st_size <= 0) {
		close(fd);
		return false;
	}
	buf.resize(st.st_size);
	ssize_t got = 0;
	while (got < st.st_size) {
		ssize_t n = read(fd, buf.data() + got, st.st_size - got);
		if (n <= 0)
			break;
		got += n;
	}
	close(fd);
	return got == st.st_size;
}

/* Ask rvd for one NV12 frame; returns width/height from the reply. */
static bool grab_frame(const char *file, int channel, int &w, int &h)
{
	char cmd[384], resp[1024];
	snprintf(cmd, sizeof(cmd), "{\"cmd\":\"save\",\"format\":\"raw\",\"file\":\"%s\",\"channel\":%d}",
		 file, channel);
	if (rss_ctrl_send_command("/var/run/rss/rvd.sock", cmd, resp, sizeof(resp), 3000) < 0) {
		fprintf(stderr, "grab: rvd not reachable\n");
		return false;
	}
	if (!strstr(resp, "\"ok\"")) {
		fprintf(stderr, "grab: %s\n", resp);
		return false;
	}
	const char *pw = strstr(resp, "\"width\":"), *ph = strstr(resp, "\"height\":");
	if (pw)
		w = atoi(pw + 8);
	if (ph)
		h = atoi(ph + 9);
	return w > 0 && h > 0;
}

/* rvd's IVS motion flag, the same poll rmd makes ("ivs-status"); -1 = no answer */
static int ivs_motion(void)
{
	char resp[512];
	if (rss_ctrl_send_command("/var/run/rss/rvd.sock", "{\"cmd\":\"ivs-status\"}", resp,
				  sizeof(resp), 1000) < 0)
		return -1;
	if (!strstr(resp, "\"active\":true"))
		return -1;
	return strstr(resp, "\"motion\":true") ? 1 : 0;
}

static const char *clock_str(void)
{
	static char b[16];
	time_t t = time(nullptr);
	strftime(b, sizeof(b), "%H:%M:%S", localtime(&t));
	return b;
}

/* ---------------- stats ---------------- */

struct Acc {
	double sum = 0, mx = 0;
	int n = 0;
	void add(double v) { sum += v, mx = std::max(mx, v), n++; }
	double avg() const { return n ? sum / n : 0; }
	void reset() { *this = Acc(); }
};

static void usage(void)
{
	fprintf(stderr,
		"usage: aitest [-b jzdl|ncnn] [-m model] [-x net_w] [-y net_h] [-i interval_ms] [-d seconds] [-r report_s]\n"
		"              [-c classes] [-t conf%%] [-C channel] [-f frame_file] [-F file_once]\n"
		"              [-p cpu] [-q] [-u (delete model files after load)] [-L (ncnn low-memory)]\n"
		"              [-g hold_s (detect only while rvd IVS reports motion, + hold_s after)]\n"
		"              [-s n (save up to n frames: detections det_*.yuv, with -g also ev_NN.yuv per event)]\n");
	exit(2);
}

int main(int argc, char **argv)
{
	const char *backend = "jzdl", *model = "/tmp/ai/magik_model_yolov5.bin";
	const char *frame_file = "/tmp/360.yuv", *once = nullptr;
	int interval = 1000, duration = 0, report = 10, ncls = -1, conf = 40, channel = 1, cpu = -1;
	int fw_once = 640, fh_once = 360, net_w = 640, net_h = 384;
	bool quiet = false, unlink_model = false, lowmem = false;
	int gate = -1, save_max = 0;
	int o;
	while ((o = getopt(argc, argv, "b:m:i:d:r:c:t:C:f:F:W:H:x:y:p:quLg:s:h")) != -1) {
		switch (o) {
		case 'b': backend = optarg; break;
		case 'm': model = optarg; break;
		case 'i': interval = atoi(optarg); break;
		case 'd': duration = atoi(optarg); break;
		case 'r': report = atoi(optarg); break;
		case 'c': ncls = atoi(optarg); break;
		case 't': conf = atoi(optarg); break;
		case 'C': channel = atoi(optarg); break;
		case 'f': frame_file = optarg; break;
		case 'F': once = optarg; break;
		case 'W': fw_once = atoi(optarg); break;
		case 'H': fh_once = atoi(optarg); break;
		case 'x': net_w = atoi(optarg); break;
		case 'y': net_h = atoi(optarg); break;
		case 'p': cpu = atoi(optarg); break;
		case 'q': quiet = true; break;
		case 'u': unlink_model = true; break;
		case 'L': lowmem = true; break;
		case 'g': gate = atoi(optarg); break;
		case 's': save_max = atoi(optarg); break;
		default: usage();
		}
	}
	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);
	setvbuf(stdout, nullptr, _IOLBF, 0);

	if (cpu >= 0) {
		cpu_set_t s;
		CPU_ZERO(&s);
		CPU_SET(cpu, &s);
		if (sched_setaffinity(0, sizeof(s), &s) != 0)
			perror("sched_setaffinity");
	}

	Backend *be = nullptr;
#ifdef AITEST_JZDL
	if (!strcmp(backend, "jzdl"))
		be = new JzdlBackend(ncls > 0 ? ncls : 4, conf / 100.f, 0.5f);
#endif
#ifdef AITEST_NCNN
	if (!strcmp(backend, "ncnn"))
		be = new NcnnBackend(net_w, net_h, ncls > 0 ? ncls : 80, conf / 100.f, 0.5f, lowmem);
#endif
	if (!be) {
		fprintf(stderr, "backend %s not built in\n", backend);
		return 2;
	}
	long rss0 = rss_kb();
	if (!be->load(model))
		return 1;
	printf("rss: %ld kB before load, %ld kB after\n", rss0, rss_kb());
	/* -u: the model files live in /tmp (tmpfs = RAM) and are now in our heap; drop the
	 * file copies so the test doesn't hold the weights twice */
	if (unlink_model) {
		char p[256];
		const char *ext[] = {"", ".param", ".bin"};
		for (const char *e : ext) {
			snprintf(p, sizeof(p), "%s%s", model, e);
			unlink(p);
		}
		printf("model files removed from tmpfs, MemAvailable %ld kB\n", mem_avail_kb());
	}

	std::vector<uint8_t> frame;
	std::vector<Det> dets;

	/* single file: run 5 times, print timings and boxes */
	if (once) {
		if (!read_frame(once, frame) || (int)frame.size() < fw_once * fh_once * 3 / 2) {
			fprintf(stderr, "cannot read %s as %dx%d NV12\n", once, fw_once, fh_once);
			return 1;
		}
		for (int k = 0; k < 5; k++) {
			double tp, ti, tq;
			be->detect(frame.data(), fw_once, fh_once, dets, tp, ti, tq);
			printf("run %d: pre %.1f ms, infer %.1f ms, post %.1f ms, %zu det\n", k, tp, ti, tq,
			       dets.size());
		}
		for (auto &d : dets)
			printf("  %s %.2f [%d,%d,%d,%d]\n", cls_name(d.cls), d.score,
			       (int)d.x1, (int)d.y1, (int)d.x2, (int)d.y2);
		printf("rss: %ld kB\n", rss_kb());
		delete be;
		return 0;
	}

	pid_t rvd = pid_of("rvd");
	printf("aitest: backend %s, frame every %d ms from rvd channel %d via %s, rvd pid %d%s\n",
	       backend, interval, channel, frame_file, (int)rvd, cpu >= 0 ? ", pinned" : "");
	if (gate >= 0) {
		int m = ivs_motion();
		if (m < 0) {
			fprintf(stderr, "gate: rvd IVS not active ([motion] enabled = true needed)\n");
			return 1;
		}
		printf("gate: detect only while IVS motion (+%d s hold), now %s\n", gate,
		       m ? "motion" : "still");
	}

	long hz = sysconf(_SC_CLK_TCK);
	Acc a_grab, a_read, a_pre, a_inf, a_post, a_total;
	int frames = 0, fails = 0, det_frames = 0, gated = 0, saved = 0;
	int ev = 0, ev_frames = 0, ev_det = 0;
	double last_motion = -1e12, ev_start = 0;
	bool active = false;
	float ev_best = 0;
	Acc a_poll;
	double t_start = now_ms(), t_rep = t_start, next = t_start;
	long my0 = proc_ticks(getpid()), rvd0 = rvd > 0 ? proc_ticks(rvd) : -1;
	long long sb0, st0;
	sys_ticks(sb0, st0);

	while (!g_stop) {
		if (gate >= 0) {
			double tg = now_ms();
			int m = ivs_motion();
			a_poll.add(now_ms() - tg);
			if (m > 0)
				last_motion = tg;
			bool want = tg - last_motion <= gate * 1000.0;
			if (want && !active) {
				ev++, ev_frames = ev_det = 0, ev_best = 0, ev_start = tg;
				printf("%s event %d: motion start\n", clock_str(), ev);
			} else if (!want && active) {
				printf("%s event %d: end after %.0f s, %d frames, %d with detections, "
				       "best %.2f\n", clock_str(), ev, (tg - ev_start) / 1000.0,
				       ev_frames, ev_det, ev_best);
			}
			active = want;
			if (!active) {
				gated++;
				goto tick;
			}
		}
		{
		double t0 = now_ms();
		int fw = 0, fh = 0;
		bool ok = grab_frame(frame_file, channel, fw, fh);
		double t1 = now_ms();
		ok = ok && read_frame(frame_file, frame) && (int)frame.size() >= fw * fh * 3 / 2;
		double t2 = now_ms();
		if (ok) {
			double tp, ti, tq;
			be->detect(frame.data(), fw, fh, dets, tp, ti, tq);
			double t3 = now_ms();
			a_grab.add(t1 - t0), a_read.add(t2 - t1), a_pre.add(tp), a_inf.add(ti), a_post.add(tq);
			a_total.add(t3 - t0);
			frames++;
			ev_frames++;
			/* -s with -g: also keep the 3rd frame of every motion event, so a miss can be
			 * checked by eye (ev_NN.yuv) */
			if (gate >= 0 && ev_frames == 3 && saved < save_max) {
				char sp[64];
				snprintf(sp, sizeof(sp), "/tmp/ai/ev_%02d.yuv", ev);
				FILE *sf = fopen(sp, "wb");
				if (sf) {
					fwrite(frame.data(), 1, (size_t)fw * fh * 3 / 2, sf);
					fclose(sf);
					saved++;
				}
			}
			if (!dets.empty()) {
				det_frames++;
				ev_det++;
				for (auto &d : dets)
					ev_best = std::max(ev_best, d.score);
				if (saved < save_max) {
					char sp[64];
					snprintf(sp, sizeof(sp), "/tmp/ai/det_%02d.yuv", saved++);
					FILE *sf = fopen(sp, "wb");
					if (sf) {
						fwrite(frame.data(), 1, (size_t)fw * fh * 3 / 2, sf);
						fclose(sf);
					}
				}
				if (!quiet) {
					printf("  %s det:", clock_str());
					for (auto &d : dets)
						printf(" %s %.2f [%d,%d,%d,%d]", cls_name(d.cls),
						       d.score, (int)d.x1, (int)d.y1, (int)d.x2, (int)d.y2);
					printf("\n");
				}
			}
		} else {
			fails++;
		}
		}
tick:
		double t = now_ms();
		if (t - t_rep >= report * 1000.0) {
			double wall = (t - t_rep) / 1000.0;
			long my1 = proc_ticks(getpid()), rvd1 = rvd > 0 ? proc_ticks(rvd) : -1;
			long long sb1, st1;
			sys_ticks(sb1, st1);
			double my_pct = 100.0 * (my1 - my0) / hz / wall;
			double rvd_pct = rvd1 >= 0 ? 100.0 * (rvd1 - rvd0) / hz / wall : -1;
			/* /proc/stat sums all CPUs; report as % of ONE core like top's per-process figure */
			double sys_pct = st1 > st0 ? 100.0 * (sb1 - sb0) / (st1 - st0) : 0;
			if (gate >= 0)
				printf("[%5.0fs] gate: %d ticks skipped (still), ivs-status poll %.1f/%.1f ms\n",
				       (t - t_start) / 1000.0, gated, a_poll.avg(), a_poll.mx);
			printf("[%5.0fs] %d frames (%.2f fps), %d fails, %d with detections | ms avg/max: "
			       "grab %.1f/%.1f read %.1f/%.1f pre %.1f/%.1f infer %.1f/%.1f post %.1f/%.1f "
			       "total %.1f/%.1f | cpu: aitest %.1f%% rvd %.1f%% system %.1f%% (all cores) | "
			       "rss %ld kB, MemAvailable %ld kB\n",
			       (t - t_start) / 1000.0, frames, frames / wall, fails, det_frames, a_grab.avg(),
			       a_grab.mx, a_read.avg(), a_read.mx, a_pre.avg(), a_pre.mx, a_inf.avg(), a_inf.mx,
			       a_post.avg(), a_post.mx, a_total.avg(), a_total.mx, my_pct, rvd_pct, sys_pct,
			       rss_kb(), mem_avail_kb());
			my0 = my1, rvd0 = rvd1, sb0 = sb1, st0 = st1, t_rep = t;
			frames = fails = det_frames = gated = 0;
			a_poll.reset();
			a_grab.reset(), a_read.reset(), a_pre.reset(), a_inf.reset(), a_post.reset(),
				a_total.reset();
		}
		if (duration > 0 && t - t_start >= duration * 1000.0)
			break;

		/* fixed-rate schedule; if we overran, start the next frame now */
		next += interval;
		double wait = next - now_ms();
		if (wait > 0)
			usleep((useconds_t)(wait * 1000));
		else
			next = now_ms();
	}
	unlink(frame_file);
	delete be;
	return 0;
}
