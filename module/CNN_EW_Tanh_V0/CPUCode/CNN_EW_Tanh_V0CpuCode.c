#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"
//typedef float real;
typedef double real;

int main(void)
{
	const int ni = 20;
	const int no = 50;
	const int row = 4;
	const int col = 4;
	const int batch_size = 384;

	int ticks = ni*no*row*col*batch_size;
	int z_offset = 0;
	int z_size = ni*no*row*col*batch_size*sizeof(real);
	int a_offset = z_size;
	int a_size = ni*no*row*col*batch_size*sizeof(real);

	real* z = (real*)malloc(z_size);
	real* a = (real*)malloc(a_size);
	
	max_file_t *maxfile = CNN_EW_Tanh_V0_init();
	max_engine_t *engine = max_load(maxfile, "*");

	printf("Writing to LMem.\n");
	max_actions_t* act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", z_offset);
	max_set_param_uint64t(act, "size", z_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", z, z_size);
	max_run(engine, act);

	printf("Running on DFE.\n");
	act = max_actions_init(maxfile, "default");
	max_set_param_uint64t(act, "ticks", ticks);
	max_set_param_uint64t(act, "input_offset", z_offset);
	max_set_param_uint64t(act, "input_size", z_size);
	max_set_param_uint64t(act, "output_offset", a_offset);
	max_set_param_uint64t(act, "output_size", a_size);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", a_offset);
	max_set_param_uint64t(act, "size", a_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", a, a_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(z);
	free(a);

	return 0;
}
