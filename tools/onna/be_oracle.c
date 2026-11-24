/* host backend: QEMU's own NNA engine code as a test oracle (oracle/mkoracle.sh). Test only.
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#include <string.h>
#include "be_oracle.h"

void *nna_oracle_new(void);
void ingenic_nna_op(void *opaque, unsigned op, uint32_t imm, uint8_t *vec);
enum { NN_RWR, NN_RRD, NN_DWR, NN_DRD, NN_MAC, NN_CMD };

static void o_rwr(void *c, unsigned reg, uint32_t v)
{
	uint8_t b[64] = { 0 };

	memcpy(b, &v, 4);
	ingenic_nna_op(c, NN_RWR, reg, b);
}

static void o_dwr(void *c, unsigned imm, const uint8_t v[64])
{
	uint8_t b[64];

	memcpy(b, v, 64);
	ingenic_nna_op(c, NN_DWR, imm, b);
}

static void o_drd(void *c, unsigned imm, uint8_t v[64])
{
	memset(v, 0, 64);
	ingenic_nna_op(c, NN_DRD, imm, v);
}

static void o_mac(void *c, unsigned imm, uint32_t word)
{
	uint8_t b[64] = { 0 };

	memcpy(b, &word, 4);
	ingenic_nna_op(c, NN_MAC, imm, b);
}

static void o_cmd(void *c, unsigned imm)
{
	uint8_t b[64] = { 0 };

	ingenic_nna_op(c, NN_CMD, imm, b);
}

void nna_be_oracle(struct nna_be *be)
{
	be->ctx = nna_oracle_new();
	be->rwr = o_rwr, be->dwr = o_dwr, be->drd = o_drd, be->mac = o_mac, be->cmd = o_cmd;
	be->direct = 0;
}
