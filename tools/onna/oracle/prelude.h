/* onna test oracle: what QEMU's ingenic-nna.c engine needs, without QEMU (host only).
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define KiB 1024
#define BIT(n) (1u << (n))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define LOG_UNIMP 1
#define LOG_GUEST_ERROR 2
#define qemu_log_mask(m, ...) fprintf(stderr, __VA_ARGS__)
#define PRIx64 "llx"
typedef uint64_t hwaddr;
static inline uint32_t extract32(uint32_t v, int s, int l) { return (v >> s) & (~0u >> (32 - l)); }
static inline uint64_t extract64(uint64_t v, int s, int l) { return (v >> s) & (~0ull >> (64 - l)); }
static inline int32_t sextract32(uint32_t v, int s, int l) { return ((int32_t)(v << (32 - l - s))) >> (32 - l); }
static inline int ctpop32(uint32_t v) { return __builtin_popcount(v); }
static inline uint32_t ldl_le_p(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline void stl_le_p(void *p, uint32_t v) { memcpy(p, &v, 4); }
static inline uint64_t ldq_le_p(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
#define trace_ingenic_nna_op(...) do { } while (0)
#define cpu_to_le32(x) (x)

#define INGENIC_NNA_WINDOW      (64 * KiB)
#define INGENIC_NNA_NNDMA_REGS  8
#define INGENIC_NNA_REGS        0x80
#define INGENIC_NNA_WRAM        (576 * KiB)
#define INGENIC_NNA_BTRAM       (16 * KiB)
#define INGENIC_NNA_FRAM        (128 * KiB)
#define INGENIC_NNA_OUT         1024
#define INGENIC_NNA_SLOTS       1024

/* the engine fields of IngenicNnaState (hw/misc/ingenic-nna.h), without the device parts */
typedef struct IngenicNnaState {
	bool t40;
	FILE *log;
	uint32_t reg[INGENIC_NNA_REGS];
	uint16_t wfcw[16];
	uint32_t wfcw_ptr, wr_w, bt_w, wr_r, wr_base;
	uint32_t port_line[2], port_half[2], port_slot[2], fline;
	int64_t sums[8 * 32], ocb[8 * 32];
	int32_t bt[64];
	bool bt_valid;
	uint8_t out[INGENIC_NNA_OUT];
	uint32_t out_len, out_rd;
	uint8_t wram[INGENIC_NNA_WRAM];
	uint8_t btram[INGENIC_NNA_BTRAM];
	uint8_t fram[INGENIC_NNA_FRAM];
	bool slots_dirty;
	uint32_t nslots[2];
	int16_t slot_inc[2][INGENIC_NNA_SLOTS];
	bool slot_jump[2][INGENIC_NNA_SLOTS];
} IngenicNnaState;
void ingenic_nna_op(void *opaque, unsigned op, uint32_t imm, uint8_t *vec);
