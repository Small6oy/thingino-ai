#!/bin/sh
# gate.sh: bit-exact gate. For each model x frame, compare libjzdl (QEMU),
# ojzdl (QEMU, mips), ojzdl (native) and jzdl_ref.py. Every output must be
# bit-identical to libjzdl's.
#   gate.sh OUTDIR MODEL.bin [MODEL.bin ...]
# Environment (defaults in brackets):
#   JZDL_QEMU   offline-runtime dir with fbuild/qemu-mipsel and run/{jzdlrun,libjzdl.m.so,*.rgb}
#               [$HOME/jzdl-qemu]; recipe: docs/jzdl-format.md "Offline runtime"
#   XB          T31 Buildroot output (sysroot) [$HOME/xb/tf-v3/output/xb]
#   REF         dir with jzdl.py + jzdl_ref.py [../jzdl next to this script]
#   PY          python with numpy [python3]
#   FRAMES      416x320 .rgb frames [the run/ frames + FRAMES_DIR/*.rgb]
#   FRAMES_DIR  extra frames [$HOME/ojzdl/frames]
set -eu
[ $# -ge 2 ] || { echo "usage: $0 OUTDIR MODEL.bin [MODEL.bin ...]" >&2; exit 2; }
OUT=$1; shift
Q=${JZDL_QEMU:-$HOME/jzdl-qemu}
XB=${XB:-$HOME/xb/tf-v3/output/xb}
S=$XB/host/mipsel-buildroot-linux-uclibc/sysroot
R=$Q/run
SRC=$(cd "$(dirname "$0")" && pwd)
REF=${REF:-$SRC/../jzdl}
PY=${PY:-python3}
FRAMES_DIR=${FRAMES_DIR:-$HOME/ojzdl/frames}
FRAMES=${FRAMES:-"$(ls "$R"/*_416x320.rgb "$FRAMES_DIR"/*.rgb 2>/dev/null | tr "\n" " ")"}
mkdir -p "$OUT"
fail=0
for m in "$@"; do
	mn=$(basename "$m" .bin)
	for fr in $FRAMES; do
		fn=$(basename "$fr" .rgb | cut -d_ -f1)
		p=$OUT/$mn.$fn
		"$Q/fbuild/qemu-mipsel" -cpu XBurstR2 -L "$S" -E LD_LIBRARY_PATH="$R" "$R/jzdlrun" "$m" "$fr" "$p.lib" >"$p.lib.log" 2>&1
		"$Q/fbuild/qemu-mipsel" -cpu XBurstR2 -L "$S" "$SRC/ojzdlrun.mips" "$m" "$fr" "$p.oq" >"$p.oq.log" 2>&1
		"$SRC/ojzdlrun" "$m" "$fr" "$p.on" >"$p.on.log" 2>&1
		(cd "$REF" && "$PY" jzdl_ref.py "$m" "$fr" "$p.ref.-999.f32" >"$p.ref.log" 2>&1)
		for k in oq on ref; do
			printf '%-12s %-5s %-4s ' "$mn" "$fn" "$k"
			"$PY" "$SRC/cmpf32.py" "$p.lib.-999.f32" "$p.$k.-999.f32" || fail=1
		done
	done
done
[ "$fail" -eq 0 ] && echo "GATE PASS" || { echo "GATE FAIL"; exit 1; }
