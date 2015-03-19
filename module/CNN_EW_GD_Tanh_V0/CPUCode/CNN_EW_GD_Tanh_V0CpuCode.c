#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"
//typedef float real;
typedef double real;

int main(void)
{
	const int ni = 2;
	const int no = 4;
	const int row = 4;
	const int col = 4;
	const int batch_size = 384;

	int ticks = ni*no*row*col*batch_size;
	int i_g_offset = 0;
	int i_g_size = ni*no*row*col*batch_size*sizeof(real);
	int i_v_offset = i_g_size;
	int i_v_size = ni*no*row*col*batch_size*sizeof(real);
	int o_offset = i_v_offset+i_v_size;
	int o_size = ni*no*row*col*batch_size*sizeof(real);

	real* a_g = (real*)malloc(i_g_size);
	real* z = (real*)malloc(i_v_size);
	real* z_g = (real*)malloc(o_size);
	
	max_file_t *maxfile = CNN_EW_GD_Tanh_V0_init();
	max_engine_t *engine = max_load(maxfile, "*");

	printf("Writing to LMem.\n");
	max_actions_t* act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", i_g_offset);
	max_set_param_uint64t(act, "size", i_g_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", a_g, i_g_size);
	max_run(engine, act);

	printf("Writing to LMem.\n");
	act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", i_v_offset);
	max_set_param_uint64t(act, "size", i_v_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", z, i_v_size);
	max_run(engine, act);

	printf("Running on DFE.\n");
	act = max_actions_init(maxfile, "default");
	max_set_param_uint64t(act, "ticks", ticks);
	max_set_param_uint64t(act, "input_value_offset", i_g_offset);
	max_set_param_uint64t(act, "input_value_size", i_g_size);
	max_set_param_uint64t(act, "input_grad_offset", i_v_offset);
	max_set_param_uint64t(act, "input_grad_size", i_v_size);
	max_set_param_uint64t(act, "output_offset", o_offset);
	max_set_param_uint64t(act, "output_size", o_size);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", o_offset);
	max_set_param_uint64t(act, "size", o_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", z_g, o_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(a_g);
	free(z);
	free(z_g);

	return 0;
}
