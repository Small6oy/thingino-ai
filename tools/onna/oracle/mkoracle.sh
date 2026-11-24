#!/bin/sh
# Build onna's host test oracle from QEMU's own NNA engine (gtxaspec/qemu hw/misc/ingenic-nna.c).
# The engine code is extracted at build time from your QEMU checkout and never copied into this
# repo: it is GPL-2.0-or-later and serves only as the reference the driver is tested against.
#   mkoracle.sh <qemu-src>/hw/misc/ingenic-nna.c <outdir>
# Output: <outdir>/nnaoracle.c (prelude + engine, lines from NNDMA_REGS to the end of
# ingenic_nna_op), then compile with: cc -O2 -c nnaoracle.c
set -eu
SRC=$1; OUT=$2
mkdir -p "$OUT"
start=$(grep -n '^#define NNDMA_REGS' "$SRC" | cut -d: -f1)
end=$(grep -n '^/\* the channels at +0x8000' "$SRC" | cut -d: -f1)
[ -n "$start" ] && [ -n "$end" ] || { echo "mkoracle: markers not found in $SRC" >&2; exit 1; }
{
	cat "$(dirname "$0")/prelude.h"
	sed -n "${start},$((end - 1))p" "$SRC"
	cat "$(dirname "$0")/epilogue.c"
} > "$OUT/nnaoracle.c"
echo "mkoracle: $OUT/nnaoracle.c from $SRC lines $start-$((end - 1))"
