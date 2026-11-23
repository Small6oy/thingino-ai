# Ingenic magik-transform-tools 2.0.4 (post-training quantisation, ONNX -> T41 NNA .bin) and
# mgkrun, our batch runner on Ingenic's PC executor (int8 simulation of the converted model).
# x86-64 only. Ingenic's files are proprietary and not part of this repo: they are downloaded at
# build time from the public magik-toolkit mirror, pinned to a commit and checked by md5.
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 python3 python3-pip python3-dev ca-certificates curl libgl1 libglib2.0-0 g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "numpy<1.24" "onnx==1.12.0" "onnxruntime==1.12.1" "protobuf<4" \
      opencv-python-headless==4.6.0.66 onnx-simplifier==0.4.8
ARG MAGIK_SRC=https://raw.githubusercontent.com/wispytrace/magik-toolkit/e511d370dd7ff84664c9140e0590c354947c7eac
ARG MAGIK=$MAGIK_SRC/TransformKit
RUN mkdir -p /opt/magik && cd /opt/magik \
    && curl -fsSL -o magik-transform-tools "$MAGIK/magik-transform-tools" \
    && curl -fsSL -o magik_transformer-2.0.4-py3-none-any.whl "$MAGIK/magik_transformer-2.0.4-py3-none-any.whl" \
    && echo "e04effb54634823d899c4914e5211bee  magik-transform-tools" | md5sum -c - \
    && echo "3e4e33bc19fee0639fabbed6ddcb9eb4  magik_transformer-2.0.4-py3-none-any.whl" | md5sum -c - \
    && chmod +x magik-transform-tools \
    && (pip3 install --no-cache-dir magik_transformer-2.0.4-py3-none-any.whl || echo "wheel install failed (non-fatal)")
# PC executor (headers + libs) and mgkrun
ARG PCI=$MAGIK_SRC/Models/post/pc_inference
COPY mgkrun.cc /opt/magik/pci/
RUN cd /opt/magik/pci && mkdir -p include lib \
    && for f in include/common.h include/graph_executor.h include/tensor.h lib/libmagikExecutor.so.2.1.0 \
                lib/libprotobuf.so.16.0.0 lib/libprotobuf-lite.so.16.0.0; do curl -fsSL -o "$f" "$PCI/$f"; done \
    && printf '%s\n' \
       "7921301eedecbeef24579d463726c0c7  include/common.h" \
       "fb1c562696b3efc4e1e947a5c42a50e7  include/graph_executor.h" \
       "d9491b96981edc3b829327c0105d292b  include/tensor.h" \
       "215967090740fa81eec1615d77284bbd  lib/libmagikExecutor.so.2.1.0" \
       "0e21e1bfa46430b13170fd6766b1e9aa  lib/libprotobuf.so.16.0.0" \
       "d4fb69e210ce75f3da72626c7cc5a6b8  lib/libprotobuf-lite.so.16.0.0" | md5sum -c - \
    && cd lib && ln -s libmagikExecutor.so.2.1.0 libmagikExecutor.so \
    && ln -s libprotobuf.so.16.0.0 libprotobuf.so.16 && ln -s libprotobuf.so.16 libprotobuf.so \
    && ln -s libprotobuf-lite.so.16.0.0 libprotobuf-lite.so.16 && ln -s libprotobuf-lite.so.16 libprotobuf-lite.so \
    && cd .. && g++ -O2 -std=c++11 -mavx -Iinclude mgkrun.cc -o /opt/magik/mgkrun -Llib \
       -lmagikExecutor -lprotobuf -lprotobuf-lite -pthread -ldl
ENV PATH=/opt/magik:$PATH LD_LIBRARY_PATH=/opt/magik/pci/lib
WORKDIR /work
