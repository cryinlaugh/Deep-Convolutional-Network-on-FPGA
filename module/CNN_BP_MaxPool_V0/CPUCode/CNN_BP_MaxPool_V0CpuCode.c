#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"
typedef unsigned char uchar;
//typedef float real;
typedef double real;

int main(void)
{
	const int no = 2;
	const int k = 2;
	const int row = 8;
	const int col = 8;
	const int batch_size = 384;

	int ticks = no*row*col*batch_size;
	int z2_offset = 0;
	int z2_size = no*row*col*batch_size*sizeof(real);
	int z_offset = z2_size;
	int z_size = no*row*col/k/k*batch_size*sizeof(real);
	int sel_offset = z_offset+z_size;
	int sel_size = no*row*col*batch_size/8;

	real* z2 = (real*)malloc(z2_size);
	real* z = (real*)malloc(z_size);
	uchar* sel = (uchar*)malloc(sel_size);
	
	max_file_t *maxfile = CNN_BP_MaxPool_V0_init();
	max_engine_t *engine = max_load(maxfile, "*");

	printf("Writing to LMem.\n");
	max_actions_t* act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", z_offset);
	max_set_param_uint64t(act, "size", z_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", z, z_size);
	max_run(engine, act);

	printf("Writing from LMemBytes.\n");
	act = max_actions_init(maxfile, "writeLMemBytes");
	max_set_param_uint64t(act, "offset", sel_offset);
	max_set_param_uint64t(act, "size", sel_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", sel, sel_size);
	max_run(engine, act);

	printf("Running on DFE.\n");
	act = max_actions_init(maxfile, "default");
	max_set_param_uint64t(act, "ticks", ticks);
	max_set_param_uint64t(act, "no", no);
	max_set_param_uint64t(act, "z_offset", z_offset);
	max_set_param_uint64t(act, "z_size", z_size);
	max_set_param_uint64t(act, "sel_offset", sel_offset);
	max_set_param_uint64t(act, "sel_size", sel_size);
	max_set_param_uint64t(act, "z2_offset", z2_offset);
	max_set_param_uint64t(act, "z2_size", z2_size);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", z2_offset);
	max_set_param_uint64t(act, "size", z2_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", z2, z2_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(z2);
	free(z);
	free(sel);

	return 0;
}
