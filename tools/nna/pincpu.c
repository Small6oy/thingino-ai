/* pincpu CPU cmd [args...]: run cmd pinned to one CPU (the camera's busybox has no taskset).
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    cpu_set_t s;

    if (argc < 3) {
        fprintf(stderr, "usage: %s cpu cmd [args...]\n", argv[0]);
        return 2;
    }
    CPU_ZERO(&s);
    CPU_SET(atoi(argv[1]), &s);
    if (sched_setaffinity(0, sizeof(s), &s)) {
        perror("sched_setaffinity");
        return 1;
    }
    execv(argv[2], argv + 2);
    perror("execv");
    return 127;
}
