#!/bin/sh
# Cross-build tools/nna/*.cpp (nnatest, nnappp) for the T41 (uClibc) against Ingenic's Magik venus
# runtime, in Docker (local, or a remote host via DOCKER_HOST / docker context).
# Needs: MAGIK = a checkout of the magik-toolkit (wispytrace/magik-toolkit), and TC_IMAGE = a Docker
# image with the Thingino xburst2 (T41) toolchain in /opt/tc (incl. /opt/tc/bin/patchelf).
#
#   tools/nna/build.sh
#   runtime: magik-toolkit InferenceKit/nna2/.../T41 (venus 0.1.2.5, matches magik-transform-tools 2.0.4)
# Output: out/<runtime>/{nnatest, lib/*.so}
set -e
cd "$(dirname "$0")"
RT=toolkit
MAGIK=${MAGIK:?set MAGIK to a magik-toolkit checkout}
TC_IMAGE=${TC_IMAGE:?set TC_IMAGE to a Docker image with the Thingino T41 toolchain in /opt/tc}
T41=$MAGIK/InferenceKit/nna2/mips720-glibc229/T41
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/include" "$STAGE/lib" "$STAGE/src"
cp -R "$T41/include/." "$STAGE/include/"
cp "$T41"/lib/uclibc/libvenus.so "$T41"/lib/uclibc/libaip.so "$T41"/lib/uclibc/libdrivers.so "$STAGE/lib/"
cp ./*.cpp "$STAGE/src/"

mkdir -p "out/$RT/lib"
COPYFILE_DISABLE=1 tar -c -C "$STAGE" include lib src | docker run --rm -i "$TC_IMAGE" sh -c '
  set -e; mkdir -p /w && tar -x -C /w 2>/dev/null; cd /w
  CXX=/opt/tc/bin/mipsel-linux-g++; P=/opt/tc/bin/mipsel-linux-
  SYSROOT=$($CXX -print-sysroot)
  for s in src/*.cpp; do
    n=$(basename $s .cpp)
    $CXX -std=c++11 -O2 -Wall -Iinclude -o $n $s -Llib -lvenus -laip -ldrivers \
      -Wl,-rpath-link,lib -Wl,-rpath,/tmp/nna/lib:/usr/lib -lpthread -lrt -lm >&2
    ${P}strip $n
  done
  mkdir -p o/lib; cp nnatest nnappp o/
  for f in lib/*.so; do ${P}strip --strip-unneeded -o o/lib/$(basename $f) $f; done
  # vendor libvenus NEEDs the loader by its old name; on uClibc 1.0.x that name is a
  # symlink to ld-uClibc.so.1 but ld.so loads it as a second copy -> jump to 0 at startup
  for f in o/lib/*.so; do
    /opt/tc/bin/patchelf --print-needed $f | grep -q "^ld-uClibc-mipsn8.so.0$" &&
      /opt/tc/bin/patchelf --replace-needed ld-uClibc-mipsn8.so.0 ld-uClibc.so.1 $f
  done
  cp -L $SYSROOT/usr/lib/libstdc++.so.6 o/lib/ 2>/dev/null || cp -L $SYSROOT/lib/libstdc++.so.6 o/lib/
  ${P}strip --strip-unneeded o/lib/libstdc++.so.6
  ${P}readelf -d o/nnatest | grep NEEDED >&2
  tar -c -C o .' | tar -x -C "out/$RT"
ls -l "out/$RT" "out/$RT/lib"
