#!/bin/sh
# Copyright (C) 2026 Thingino Project
# SPDX-License-Identifier: MIT
# v4run.sh CAMERA MODEL.bin FRAME.rgb [GRAPH|-] [N]: onnaq.mips -e nna -T -n N on a T41 camera, safely.
#
# Two processes driving the NNA at once wedge the T41 (watchdog reset). raptor's S31raptor runs a
# watcher that restarts the whole stack within ~5 s of `raptorctl rvd stop`, so this stops the
# watcher first, then rvd (whose IVS backend may use the NNA), polls that rvd stays down during the
# run, and restores everything with `S31raptor restart` (also on failure). Files go to /tmp/onnav4
# on the camera and are removed. Prints onnaq's timing (-T per op kind) and the head md5s.
#   CAMERA      root@host
#   SSHPASS     the camera password, if it has no key (used with sshpass -e; never stored)
#   ONNAQ_MIPS  onnaq.mips [onnaq.mips next to this script] (make onnaq.mips)
set -eu
[ $# -ge 3 ] || { sed -n '4,14p' "$0"; exit 2; }
cam=$1 bin=$2 frame=$3 graph=${4:--} n=${5:-20}
here=$(cd "$(dirname "$0")" && pwd)
ONNAQ_MIPS=${ONNAQ_MIPS:-$here/onnaq.mips}
if [ -n "${SSHPASS:-}" ]; then SSH="sshpass -e ssh"; SCP="sshpass -e scp -O"; else SSH=ssh; SCP="scp -O"; fi
D=/tmp/onnav4
files="$ONNAQ_MIPS $bin $frame"
[ "$graph" = - ] || files="$files $graph"
$SSH "$cam" "rm -rf $D && mkdir -p $D"
$SCP -q $files "$cam:$D/"
g=-; [ "$graph" = - ] || g=$D/$(basename "$graph")
$SSH "$cam" "
	restore() { kill \$(cat $D/poll.pid 2>/dev/null) 2>/dev/null; /etc/init.d/S31raptor restart >/dev/null 2>&1; rm -rf $D; }
	trap restore EXIT
	start-stop-daemon -K -q -p /var/run/rss/watch.pid 2>/dev/null; rm -f /var/run/rss/watch.pid
	raptorctl rvd stop >/dev/null 2>&1; sleep 4
	pidof rvd >/dev/null && { echo 'rvd still running: not starting'; exit 1; }
	(while :; do pidof rvd >/dev/null && echo \$(date +%T) rvd running; sleep 1; done) > $D/poll.log 2>&1 &
	echo \$! > $D/poll.pid
	cd $D && ./$(basename "$ONNAQ_MIPS") -e nna -T -n $n $(basename "$bin") $g $(basename "$frame") out 2>&1 | grep -v 'W blocks'
	for f in out.*.f32; do echo \"\$f \$(md5sum \$f | cut -c1-32)\"; done
	echo \"rvd-running samples during the run: \$(grep -c running $D/poll.log)\"
"
echo "restored: $($SSH "$cam" 'pidof rvd >/dev/null && echo rvd up || echo rvd DOWN; ps | grep -c "[S]31raptor watch" | sed "s/^/watchers /"')"
