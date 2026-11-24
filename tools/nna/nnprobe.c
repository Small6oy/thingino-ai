/*
 * nnprobe — which MXU3 / NN instructions does each T41 core accept?
 *
 *   nnprobe            run every probe on CPU0 and on CPU1 (forks one child per core)
 *
 * Each probe executes one instruction word under sigsetjmp and reports OK or SIGILL.
 * The first MXU3 instruction traps CU2-unusable; the kernel then sets ST0_CU2 for the
 * thread (arch/mips/kernel/traps.c), so that one is retried transparently.
 *
 * The NN words write into the NNA's own W/F RAM or registers only (no memory side
 * effects); run it only while nothing else uses the NNA.
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static sigjmp_buf jb;
static void on_sig(int s) { siglongjmp(jb, s); }

#define PROBE(name, word)                                                                      \
    static void name(void) { __asm__ __volatile__(".set push\n.set noreorder\n.word " #word   \
                                                  "\nnop\n.set pop" ::: "memory", "t0"); }

/* MXU3 control: CFCMXU t0, mir (from the vendor kernel's mxuv3.c) */
PROBE(p_cfcmxu_mir, 0x70100203)
PROBE(p_cfcmxu_mcsr, 0x70100a03)
/* the exact NNDWR words libvenus runs (vr0..vr3, imm from bits 20:16) */
PROBE(p_nndwr_vr0, 0x7041003a)
PROBE(p_nndwr_vr1, 0x7041083a)
PROBE(p_nndwr_vr2, 0x7041103a)
PROBE(p_nndwr_vr3, 0x7041183a)
/* other NN classes, immediates 0 */
PROBE(p_nnrrd_0, 0x7020003a)
PROBE(p_nndrd_0, 0x7060003a)
PROBE(p_nnrwr_0, 0x7000003a)

static const struct {
    const char *name;
    void (*fn)(void);
} probes[] = {
    {"CFCMXU mir   0x70100203", p_cfcmxu_mir},  {"CFCMXU mcsr  0x70100a03", p_cfcmxu_mcsr},
    {"NNRRD  0     0x7020003a", p_nnrrd_0},     {"NNDRD  0     0x7060003a", p_nndrd_0},
    {"NNRWR  0     0x7000003a", p_nnrwr_0},     {"NNDWR  vr0   0x7041003a", p_nndwr_vr0},
    {"NNDWR  vr1   0x7041083a", p_nndwr_vr1},   {"NNDWR  vr2   0x7041103a", p_nndwr_vr2},
    {"NNDWR  vr3   0x7041183a", p_nndwr_vr3},
};

static void run_on(int cpu)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof(s), &s)) {
        perror("sched_setaffinity");
        exit(1);
    }
    signal(SIGILL, on_sig);
    signal(SIGSEGV, on_sig);
    signal(SIGBUS, on_sig);
    for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        int sig = sigsetjmp(jb, 1);
        if (!sig) {
            probes[i].fn();
            printf("cpu%d  %-24s OK   (ran on cpu%d)\n", cpu, probes[i].name, sched_getcpu());
        } else {
            printf("cpu%d  %-24s %s\n", cpu, probes[i].name, strsignal(sig));
        }
        fflush(stdout);
    }
}

int main(void)
{
    for (int cpu = 0; cpu < 2; cpu++) {
        pid_t p = fork();
        if (!p) {
            run_on(cpu);
            _exit(0);
        }
        int st;
        waitpid(p, &st, 0);
    }
    return 0;
}
