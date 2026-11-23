// SPDX-License-Identifier: GPL-2.0
/*
 * t41l2 — give the T41 NNA its 384 KiB of ORAM at runtime (L2 256 -> 128 KiB)
 *
 * A T41NQ boots with 256 KiB of its 512 KiB on-chip SRAM as L2 (CCU MSCR
 * bits 12:10 = 2), which puts the ORAM at 0x12640000; the vendor soc-nna
 * driver only maps it at 0x12620000 and wedges a venus program in its mmap.
 * The fix for production is in U-Boot (CONFIG_T41_L2_128K); this module does
 * the same switch from a running kernel, so it can be tried on a camera
 * without flashing a boot loader. Nothing persists: a reboot undoes it.
 *
 * The L2 lines the switch gives up are lost with whatever they hold, so the
 * switch runs under stop_machine(): every CPU writes its L1 back, and the
 * CPU doing the switch then writes back the whole L2 and sets MSCR with no
 * memory store in between. The CPUs hand off through flags they only touch
 * through KSEG1 (uncached), so the handshake leaves no dirty line behind.
 *
 * Linux keeps the L2 size it probed at boot (256 KiB) for its whole-cache
 * index loops; on a 128 KiB L2 those just visit each line twice.
 *
 *   insmod t41l2.ko            switch to 128 KiB (share=1)
 *   dmesg | grep t41l2
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/smp.h>
#include <linux/stop_machine.h>
#include <linux/cpumask.h>
#include <asm/addrspace.h>
#include <asm/cacheops.h>
#include <asm/mipsregs.h>
#include <asm/r4kcache.h>
#include <asm/io.h>

#define CCU_MSCR	((void __iomem *)0xb2200060)
#define MSCR_SHARE_SHIFT	10
#define MSCR_SHARE_MASK		(7u << MSCR_SHARE_SHIFT)

#define XB2_L1D_SIZE	(32 * 1024)	/* 8 ways x 128 sets x 32 B */
#define XB2_L1_LINE	32

static unsigned int share = 1;		/* 1 = 128 KiB of L2 */
module_param(share, uint, 0444);
MODULE_PARM_DESC(share, "L2 share in MSCR[12:10]: 1=128K 2=256K (default 1)");

struct t41l2_ctx {
	volatile u32 *flags;	/* KSEG1 alias: [cpu] = L1 written back, [NR_CPUS] = done */
	unsigned int master;
	u32 new_mscr;
	u32 sets, line, ways;
};

/* Leaf, registers only: no store reaches memory except the MSCR write. */
static noinline void t41l2_l1_writeback(void)
{
	unsigned long a;

	for (a = CKSEG0; a < CKSEG0 + XB2_L1D_SIZE; a += XB2_L1_LINE)
		cache_op(Index_Writeback_Inv_D, a);
	__sync();
}

static noinline void t41l2_l2_switch(u32 sets, u32 line, u32 ways, u32 new_mscr)
{
	unsigned long a, end = CKSEG0 + (unsigned long)sets * line * ways;

	t41l2_l1_writeback();
	if (line)
		for (a = CKSEG0; a < end; a += line)
			cache_op(Index_Writeback_Inv_SD, a);
	__sync();
	(void)*(volatile u32 *)CKSEG1;		/* drain the write buffer */
	__raw_writel(new_mscr, CCU_MSCR);
	__sync();
	(void)__raw_readl(CCU_MSCR);
}

static int t41l2_stop(void *data)
{
	struct t41l2_ctx *c = data;
	unsigned int cpu = smp_processor_id(), i;

	if (cpu != c->master) {
		t41l2_l1_writeback();
		c->flags[cpu] = 1;			/* uncached */
		while (!c->flags[NR_CPUS])		/* uncached reads only */
			cpu_relax();
		return 0;
	}

	for_each_online_cpu(i)
		if (i != cpu)
			while (!c->flags[i])
				cpu_relax();
	t41l2_l2_switch(c->sets, c->line, c->ways, c->new_mscr);
	c->flags[NR_CPUS] = 1;
	return 0;
}

static int __init t41l2_init(void)
{
	u32 mscr = __raw_readl(CCU_MSCR), cur = (mscr & MSCR_SHARE_MASK) >> MSCR_SHARE_SHIFT;
	u32 c2 = read_c0_config2();
	struct t41l2_ctx c = {
		.sets = 64u << ((c2 >> 8) & 0xf),
		.line = ((c2 >> 4) & 0xf) ? 2u << ((c2 >> 4) & 0xf) : 0,
		.ways = (c2 & 0xf) + 1,
	};
	void *buf;
	int ret;

	if (share < 1 || share > 4)
		return -EINVAL;
	pr_info("t41l2: MSCR 0x%08x (L2 %u KiB), Config2 0x%08x (%u sets x %u B x %u ways)\n",
		mscr, cur ? 128u << (cur - 1) : 0, c2, c.sets, c.line, c.ways);
	if (cur == share) {
		pr_info("t41l2: already %u KiB of L2, nothing to do\n", 128u << (share - 1));
		return 0;
	}

	/* lowmem, so it has a KSEG1 alias; no cached copy may exist */
	buf = kzalloc(L1_CACHE_BYTES + (NR_CPUS + 1) * sizeof(u32), GFP_KERNEL);
	if (!buf)
		return -ENOMEM;
	dma_cache_wback_inv((unsigned long)buf, L1_CACHE_BYTES + (NR_CPUS + 1) * sizeof(u32));
	c.flags = (volatile u32 *)CKSEG1ADDR(buf);
	c.master = cpumask_first(cpu_online_mask);
	c.new_mscr = (mscr & ~MSCR_SHARE_MASK) | (share << MSCR_SHARE_SHIFT);

	ret = stop_machine(t41l2_stop, &c, cpu_online_mask);
	kfree(buf);

	mscr = __raw_readl(CCU_MSCR);
	cur = (mscr & MSCR_SHARE_MASK) >> MSCR_SHARE_SHIFT;
	pr_info("t41l2: now MSCR 0x%08x: L2 %u KiB, NNA ORAM %u KiB from 0x%08x (stop_machine %d)\n",
		mscr, 128u << (cur - 1), 512 - (128u << (cur - 1)),
		0x12600000 + (128u << (cur - 1)) * 1024, ret);
	return ret;
}

static void __exit t41l2_exit(void)
{
	/* nothing to undo: the split stays until the next reboot */
}

module_init(t41l2_init);
module_exit(t41l2_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("T41: shrink the L2 to give the NNA its ORAM");
