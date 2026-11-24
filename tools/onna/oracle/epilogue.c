
/* onna oracle API */
#include <stdlib.h>
void *nna_oracle_new(void)
{
	IngenicNnaState *s = calloc(1, sizeof(*s));

	nn_reset_engine(s);
	return s;
}
