/* setaff pid mask: sched_setaffinity for any task (test tool; the camera has no taskset)
 *
 * Copyright (C) 2026 Thingino Project
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	cpu_set_t cs;
	unsigned long m;
	int pid;

	if (argc != 3)
		return 2;
	pid = atoi(argv[1]);
	m = strtoul(argv[2], 0, 0);
	CPU_ZERO(&cs);
	for (int i = 0; i < 2; i++)
		if (m >> i & 1)
			CPU_SET(i, &cs);
	if (sched_setaffinity(pid, sizeof(cs), &cs)) {
		perror("sched_setaffinity");
		return 1;
	}
	CPU_ZERO(&cs);
	sched_getaffinity(pid, sizeof(cs), &cs);
	printf("pid %d affinity now %d%d\n", pid, CPU_ISSET(1, &cs), CPU_ISSET(0, &cs));
	return 0;
}
