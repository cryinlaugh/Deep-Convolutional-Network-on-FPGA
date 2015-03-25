#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"
//typedef float real;
typedef double real;
typedef int cateType;

int main(void)
{
	const int ni = 5;
	const int no = 10;
	const int batch_size = 384;

	const int x_offset = 0;
	const int x_size = ni*batch_size*sizeof(real);
	const int z_offset = x_size;
	const int z_size = no*batch_size*sizeof(real);
	const int sm_offset = z_offset+z_size;
	const int sm_size = no*batch_size*sizeof(real);
	const int w_size = ni*no*sizeof(real);
	const int b_size = no*sizeof(real);
	const int pred_size = batch_size*sizeof(cateType);

	real* x = (real*)malloc(x_size);
	real* w = (real*)malloc(w_size);
	real* b = (real*)malloc(b_size);
	real* z = (real*)malloc(z_size);
	real* sm = (real*)malloc(sm_size);
	cateType* pred = (cateType*)malloc(pred_size);
	
	max_file_t *maxfile = CNN_FW_Softmax_V1_init();
	max_engine_t *engine = max_load(maxfile, "*");
	max_actions_t* act;

	printf("Writing to LMem.\n");
	act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", x_offset);
	max_set_param_uint64t(act, "size", x_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", x, x_size);
	max_run(engine, act);

	printf("Running on DFE.\n");
	act = max_actions_init(maxfile, "default");
	max_set_param_uint64t(act, "ni", ni);
	max_set_param_uint64t(act, "x_offset", x_offset);
	max_queue_input(act, "w", w, w_size);
	max_queue_input(act, "b", b, b_size);
	max_set_param_uint64t(act, "z_offset", z_offset);
	max_set_param_uint64t(act, "softmax_offset", sm_offset);
	max_queue_output(act, "pred", pred, pred_size);
	max_run(engine, act);

	printf("Reading from LMem.\n");
	act = max_actions_init(maxfile, "readLMem");
	max_set_param_uint64t(act, "offset", sm_offset);
	max_set_param_uint64t(act, "size", sm_size);
	max_queue_output(act, "lmem_to_cpu_at_cpu", sm, sm_size);
	max_run(engine, act);

	max_unload(engine);
	printf("Done.\n");

	free(x);
	free(w);
	free(b);
	free(z);
	free(sm);
	free(pred);

	return 0;
}
