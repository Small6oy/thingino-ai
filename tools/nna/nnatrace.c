/*
 * nnatrace.so — LD_PRELOAD logger for what the venus/libdrivers runtime asks of the kernel.
 *
 *   LD_PRELOAD=/path/nnatrace.so ./nnatest ...
 *
 * Logs (unbuffered, to stderr) every open of /dev and /proc files, every mmap/munmap, and every ioctl
 * on those device fds with the first 32 bytes of the argument before and after the call.
 * Written to find which soc-nna mmap() hits BUG_ON(!pte_none) in remap_pfn_range.
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

static char devname[64][32];

static void logf_(const char *fmt, ...)
{
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    if (n > 0)
        write(2, b, n < (int)sizeof(b) ? n : (int)sizeof(b) - 1);
}

/* name of a device fd, also for fds opened behind our back (fopen uses uClibc's internal open) */
static const char *fdname(int fd)
{
    if (fd < 0 || fd >= 64)
        return NULL;
    if (!devname[fd][0]) {
        char p[32], t[32];
        snprintf(p, sizeof(p), "/proc/self/fd/%d", fd);
        ssize_t n = readlink(p, t, sizeof(t) - 1);
        if (n <= 0)
            return NULL;
        t[n] = 0;
        if (strncmp(t, "/dev/", 5))
            return NULL;
        memcpy(devname[fd], t, n + 1);
    }
    return devname[fd];
}

int open(const char *path, int flags, ...)
{
    static int (*real)(const char *, int, ...);
    if (!real)
        real = dlsym(RTLD_NEXT, "open");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    int fd = real(path, flags, mode);
    if (!strncmp(path, "/dev/", 5) || !strncmp(path, "/proc/", 6)) {
        logf_("[trace] open(%s, 0x%x) = %d\n", path, flags, fd);
        if (fd >= 0 && fd < 64)
            snprintf(devname[fd], sizeof(devname[fd]), "%s", path);
    }
    return fd;
}

int close(int fd)
{
    static int (*real)(int);
    if (!real)
        real = dlsym(RTLD_NEXT, "close");
    if (fdname(fd)) {
        logf_("[trace] close(%s)\n", fdname(fd));
        devname[fd][0] = 0;
    }
    return real(fd);
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real)
        real = dlsym(RTLD_NEXT, "mmap");
    if (fdname(fd))
        logf_("[trace] mmap(%s, off=0x%08lx len=0x%x (%u KiB) end=0x%08lx prot=%d flags=0x%x) ...\n",
              fdname(fd), (unsigned long)off, (unsigned)len, (unsigned)(len >> 10),
              (unsigned long)off + len, prot, flags);
    void *r = real(addr, len, prot, flags, fd, off);
    if (fdname(fd))
        logf_("[trace]   = %p\n", r);
    return r;
}

int munmap(void *addr, size_t len)
{
    static int (*real)(void *, size_t);
    if (!real)
        real = dlsym(RTLD_NEXT, "munmap");
    return real(addr, len);
}

static void hex32(const char *tag, const void *p)
{
    const unsigned *u = p;
    logf_("[trace]   %s %08x %08x %08x %08x %08x %08x %08x %08x\n", tag, u[0], u[1], u[2], u[3],
          u[4], u[5], u[6], u[7]);
}

int ioctl(int fd, unsigned long req, ...)
{
    static int (*real)(int, unsigned long, ...);
    if (!real)
        real = dlsym(RTLD_NEXT, "ioctl");
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    const char *n = fdname(fd);
    /* only dump args that look like user pointers */
    int ptr = n && (unsigned long)arg > 0x10000 && (unsigned long)arg < 0x80000000UL;
    if (n) {
        logf_("[trace] ioctl(%s, 0x%08lx, %p)\n", n, req, arg);
        if (ptr)
            hex32("in ", arg);
    }
    int r = real(fd, req, arg);
    if (n) {
        if (ptr)
            hex32("out", arg);
        logf_("[trace]   = %d\n", r);
    }
    return r;
}
