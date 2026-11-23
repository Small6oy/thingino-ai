#!/bin/sh
# Prepare the ncnn CPU backend for aitest, on a Linux build host, all in Docker
# (needs a local image `thingino-yolo-export` with onnx + onnx-simplifier, see 1a below):
#  1. cut our yolov5n ONNX at the three Detect convs (the same graph Magik converts for
#     the NNA) and convert it with ncnn's onnx2ncnn -> yolov5n.param/.bin
#  2. cross-build libncnn.a, plain C++ (no MSA/MMI/OpenMP/threads), with each camera's
#     own Buildroot toolchain: tf-v4 (T41, xburst2) and tf-v3 (T31, xburst1)
#
#   tools/aitest/ncnn.sh            -> ~/aitest/ncnn/{model,v4,v3} on the build host
#   model/yolov5n.{param,bin}: any input size (aitest -x/-y); 640x384 = the NNA model
set -eu
HOST=${BUILD_HOST:?set BUILD_HOST to your Linux build host (an ssh name)}
NCNN_TAG=20240820
ONNX=${YOLOV5N_ONNX:?set YOLOV5N_ONNX to the yolov5n.onnx path on the build host}

ssh "$HOST" "NCNN_TAG=$NCNN_TAG ONNX=$ONNX sh -s" <<'EOF'
set -eu
W=$HOME/aitest/ncnn
mkdir -p "$W/model"
[ -d "$W/src" ] || git clone -q --depth 1 -b "$NCNN_TAG" https://github.com/Tencent/ncnn "$W/src"

# 1a. cut the ONNX at the Detect convs. The two upsample Resizes get their output size
#     from Shape/Slice/Concat, which ncnn can't run; give them constant scales [1,1,2,2]
#     instead (same result, any input size) and prune the dead Shape chain.
#     (onnxsim would fold it too, but onnxruntime won't load on the build host.)
docker run --rm -u "$(id -u):$(id -g)" -v "$W/model:/o" -v "$ONNX:/in.onnx:ro" thingino-yolo-export python3 -c '
import onnx, onnx.utils
from onnx import helper, TensorProto
outs = ["/model.24/m.%d/Conv_output_0" % i for i in range(3)]
onnx.utils.extract_model("/in.onnx", "/o/cut.onnx", ["images"], outs)
m = onnx.load("/o/cut.onnx")
g = m.graph
g.initializer.append(helper.make_tensor("up2_scales", TensorProto.FLOAT, [4], [1, 1, 2, 2]))
g.initializer.append(helper.make_tensor("up2_roi", TensorProto.FLOAT, [0], []))
for n in g.node:
	if n.op_type == "Resize":
		del n.input[:]
		n.input.extend([n.name.replace("/Resize", "") + "_in", "up2_roi", "up2_scales"])
# the Resize data input was renamed above only to find it again; restore it
prod = {o: n for n in g.node for o in n.output}
for n in g.node:
	if n.op_type == "Resize":
		blk = n.name.split("/")[1]            # model.11 / model.15
		src = {"model.11": "/model.10/act/Mul_output_0", "model.15": "/model.14/act/Mul_output_0"}[blk]
		n.input[0] = src
# prune nodes whose outputs nobody uses (the Shape chain)
while True:
	used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
	dead = [n for n in g.node if not any(o in used for o in n.output)]
	if not dead:
		break
	for n in dead:
		g.node.remove(n)
# input size free: ncnn ignores it, but keep the file honest
for d in g.input[0].type.tensor_type.shape.dim[2:]:
	d.dim_param = "h" if not d.dim_param and d.dim_value == 384 else "w"
onnx.checker.check_model(m)
onnx.save(m, "/o/yolov5n_cut.onnx")
from collections import Counter
print("cut:", dict(Counter(n.op_type for n in g.node)))
'

# 1b + 2. tools and cross builds in one Ubuntu 24.04 container (glibc >= 2.38, as the xb toolchains need); the xb toolchains are
# mounted at their own paths (Buildroot wrappers resolve relative to themselves)
docker run --rm -v "$W:/w" -v "$HOME/xb:$HOME/xb:ro" -e XB="$HOME/xb" -e OWN="$(id -u):$(id -g)" ubuntu:24.04 sh -ec '
trap "chown -R \$OWN /w" EXIT
apt-get -qq update >/dev/null
apt-get -qq install -y cmake g++ make libprotobuf-dev protobuf-compiler >/dev/null
# host onnx2ncnn
cmake -S /w/src -B /w/host -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF \
	-DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
	-DNCNN_BUILD_TESTS=OFF >/dev/null
make -C /w/host -j6 onnx2ncnn >/dev/null
/w/host/tools/onnx/onnx2ncnn /w/model/yolov5n_cut.onnx /w/model/yolov5n.param /w/model/yolov5n.bin
grep -c -w -E "Shape|Slice|Cast" /w/model/yolov5n.param || echo "no Shape/Slice/Cast layers"
ls -l /w/model
# target libs: plain C++, single thread, no SIMD (T41 has only mxuv3, T31 only MXU1)
for cam in v4 v3; do
	TC=$XB/tf-$cam/output/xb/host/bin/mipsel-linux-
	cmake -S /w/src -B /w/b-$cam -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=mips \
		-DCMAKE_C_COMPILER=${TC}gcc -DCMAKE_CXX_COMPILER=${TC}g++ \
		-DNCNN_MSA=OFF -DNCNN_MMI=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF \
		-DNCNN_RUNTIME_CPU=OFF -DNCNN_VULKAN=OFF -DNCNN_SIMPLEOCV=ON \
		-DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
		-DNCNN_BUILD_TESTS=OFF -DNCNN_SHARED_LIB=OFF \
		-DCMAKE_INSTALL_PREFIX=/w/$cam >/dev/null
	make -C /w/b-$cam -j6 >/dev/null
	make -C /w/b-$cam install >/dev/null
	ls -l /w/$cam/lib/libncnn.a
done
'
EOF
