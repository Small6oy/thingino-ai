// mgkdump MODEL.mgk H W FRAME.rgb OUTDIR name... : run one frame, write OUTDIR/<name>.f32 (+ shape)
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
    const int h = atoi(argv[2]), w = atoi(argv[3]), c = 3, fsz = h * w * c;
    std::ifstream in(argv[4], std::ios::binary);
    std::vector<unsigned char> buf(fsz);
    in.read((char *)buf.data(), fsz);
    GraphExecutor ex{std::string(argv[1])};
    ex.set_inplace(false);
    Tensor *t = new Tensor({1, h, w, c}, Tensor::DataType::DT_FLOAT);
    t->set_name(ex.get_input_names()[0]);
    for (int i = 0; i < fsz; i++) t->mutable_data<float>(i) = (float)buf[i];
    ex.set_input(t);
    ex.work();
    for (int a = 6; a < argc; a++) {
        const Tensor *r = ex.get_node_tensor(argv[a]);
        if (!r) { printf("%s: none\n", argv[a]); continue; }
        std::vector<float> v = ex.get_node_float_data(r);
        std::ofstream o(std::string(argv[5]) + "/" + argv[a] + ".f32", std::ios::binary);
        o.write((const char *)v.data(), v.size() * sizeof(float));
        printf("%s", argv[a]);
        for (int d : r->shape()) printf(" %d", d);
        printf(" n=%zu\n", v.size());
    }
    return 0;
}
