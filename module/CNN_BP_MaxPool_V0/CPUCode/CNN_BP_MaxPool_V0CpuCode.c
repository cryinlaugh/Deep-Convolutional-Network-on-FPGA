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

	int a_grad_offset = 0;
	int a_grad_size = no*row*col/k/k*batch_size*sizeof(real);
	int z_offset = a_grad_size;
	int z_size = no*row*col/k/k*batch_size*sizeof(real);
	int sel_offset = z_offset+z_size;
	int sel_size = no*row*col*batch_size/8;
	int z2_grad_offset = sel_offset+sel_size;
	int z2_grad_size = no*row*col*batch_size*sizeof(real);

	real* a_grad = (real*)malloc(a_grad_size);
	real* z = (real*)malloc(z_size);
	uchar* sel = (uchar*)malloc(sel_size);
	real* z2_grad = (real*)malloc(z2_grad_size);
	
	max_file_t *maxfile = CNN_BP_MaxPool_V0_init();
	max_engine_t *engine = max_load(maxfile, "*");
	max_actions_t* act;

	printf("Writing to LMem.\n");
	act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", a_grad_offset);
	max_set_param_uint64t(act, "size", a_grad_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", a_grad, a_grad_size);
	max_run(engine, act);

	printf("Writing to LMem.\n");
	act = max_actions_init(maxfile, "writeLMem");
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
	max_set_param_uint64t(act, "no", no);
	max_set_param_uint64t(act, "a_grad_offset", a_grad_offset);
	max_set_param_uint64t(act, "z_offset", z_offset);
	max_set_param_uint64t(act, "sel_offset", sel_offset);
	max_set_param_uint64t(act, "z2_grad_offset", z2_grad_offset);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", z2_grad_offset);
	max_set_param_uint64t(act, "size", z2_grad_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", z2_grad, z2_grad_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(a_grad);
	free(z);
	free(sel);
	free(z2_grad);

	return 0;
}
