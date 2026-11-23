#!/bin/sh
# Cross-build aitest on a Linux build host with each camera's Thingino Buildroot output
# (~/xb/tf-v3, ~/xb/tf-v4 there; Ingenic's sample models in ~/aitest/magik_models),
# against raptor's librss_ipc (control socket). Run tools/aitest/ncnn.sh once first.
#
#   tools/aitest/build.sh          -> out/v4/ and out/v3/ (copy one to /tmp/ai on the camera)
#
#   v4 (T41, mxuv3 only): ncnn backend; libstdc++ linked statically (the image has none).
#      Ingenic's JZDL can't run here: its 7.2.0 build is MSA, its 5.4.0 build MXU1.
#   v3 (T31, MXU1):       ncnn + jzdl backends; libjzdl.m.so (5.4.0, uClibc, MXU1) needs
#      the shared libstdc++, which is shipped (stripped, ~2 MB).
#
# /tmp is RAM: run aitest with -u (deletes the model files once loaded) and remove /tmp/ai
# afterwards.
set -eu
cd "$(dirname "$0")"
HOST=${BUILD_HOST:?set BUILD_HOST to your Linux build host (an ssh name)}
# the source goes to ~/aitest/aitest.cpp on the build host
ssh "$HOST" "mkdir -p aitest" && scp -q aitest.cpp "$HOST:aitest/aitest.cpp"
SRC=aitest

ssh "$HOST" "SRC=$SRC sh -s" <<'EOF'
set -eu
N=$HOME/aitest/ncnn
JM="$HOME/aitest/magik_models/Txx/jzdl/magik_model_yolov5.bin $HOME/aitest/magik_models/Txx/jzdl/magik_model_persondet.bin"
for cam in v4 v3; do
	B=$HOME/xb/tf-$cam/output/xb
	S=$B/host/mipsel-buildroot-linux-uclibc/sysroot
	TC=$B/host/bin/mipsel-linux-
	O=$HOME/aitest/build/$cam
	rm -rf "$O"
	mkdir -p "$O"
	cp "$N/model/yolov5n.param" "$N/model/yolov5n.bin" "$O/"
	FLAGS="-std=c++11 -O2 -Wall -Wno-unused-result -DAITEST_NCNN -I$N/$cam/include"
	LIBS="$N/$cam/lib/libncnn.a -L$S/usr/lib -lrss_ipc -lrss_common -lpthread -lm"
	if [ "$cam" = v3 ]; then
		JL=$(ls -d "$B"/build/ingenic-lib-*/acceleration-modules/jzdl/lib/5.4.0/uclibc | head -n 1)
		JI=$(ls -d "$B"/build/thingino-raptor-hal-*/ingenic-headers/Txx/jzdl | head -n 1)
		cp "$JL/libjzdl.m.so" $JM "$O/"
		"${TC}strip" --strip-unneeded -o "$O/libstdc++.so.6" "$S/usr/lib/libstdc++.so.6"
		FLAGS="$FLAGS -DAITEST_JZDL -DJZ_MXU=0 -I$JI"
		LIBS="-L$O -l:libjzdl.m.so $LIBS -Wl,-rpath,/tmp/ai"
	else
		LIBS="$LIBS -static-libstdc++ -static-libgcc"
	fi
	# shellcheck disable=SC2086 # FLAGS/LIBS are word lists
	"${TC}g++" $FLAGS -o "$O/aitest" "$SRC/aitest.cpp" $LIBS
	"${TC}strip" "$O/aitest"
	echo "== $cam"
	"${TC}readelf" -d "$O/aitest" | grep NEEDED
	ls -l "$O"
done
EOF
rm -rf out
mkdir -p out
scp -q -r "$HOST:aitest/build/v4" "$HOST:aitest/build/v3" out/
du -sh out/v4 out/v3
