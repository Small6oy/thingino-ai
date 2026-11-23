// mgkrun: run a converted T41 model (save-magik/model_quant.mgk) on the PC with Ingenic's
// magikExecutor (int8 simulation of the NNA), for many frames in one process.
//
//   mgkrun MODEL.mgk H W FRAMES.u8 OUTPREFIX
//
// FRAMES.u8: N frames of H x W x 3 uint8 RGB (raw pixels; the model holds the normalisation).
// Writes OUTPREFIX.<output>.f32 (N x tensor floats, the executor's NHWC layout) for every model
// output and prints one line per output: "<name> <dim0> <dim1> ...".
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "graph_executor.h"
#include "tensor.h"

using magik::transformkit::magikexecutor::GraphExecutor;
using magik::transformkit::magikexecutor::Tensor;

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL.mgk H W FRAMES.u8 OUTPREFIX\n", argv[0]);
        return 2;
    }
    const int h = atoi(argv[2]), w = atoi(argv[3]), c = 3, fsz = h * w * c;
    std::ifstream in(argv[4], std::ios::binary | std::ios::ate);
    if (!in) { fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
    const long n = (long)in.tellg() / fsz;
    in.seekg(0);
    GraphExecutor ex{std::string(argv[1])};
    ex.set_inplace(false);
    std::vector<std::string> outs = ex.get_output_names();
    std::vector<std::ofstream> files;
    for (auto &o : outs) {
        std::string safe = o;
        for (auto &ch : safe) if (ch == '/' || ch == ':') ch = '_';
        files.emplace_back(std::string(argv[5]) + "." + safe + ".f32", std::ios::binary);
    }
    std::vector<unsigned char> buf(fsz);
    for (long k = 0; k < n; k++) {
        in.read((char *)buf.data(), fsz);
        Tensor *t = new Tensor({1, h, w, c}, Tensor::DataType::DT_FLOAT);
        t->set_name(ex.get_input_names()[0]);
        for (int i = 0; i < fsz; i++) t->mutable_data<float>(i) = (float)buf[i];
        ex.set_input(t);
        ex.work();
        for (size_t j = 0; j < outs.size(); j++) {
            const Tensor *r = ex.get_node_tensor(outs[j]);
            std::vector<float> v = ex.get_node_float_data(r);
            files[j].write((const char *)v.data(), v.size() * sizeof(float));
            if (k == 0) {
                printf("%s", outs[j].c_str());
                for (int d : r->shape()) printf(" %d", d);
                printf("\n");
            }
        }
        delete t;
    }
    fprintf(stderr, "%ld frames\n", n);
    return 0;
}
