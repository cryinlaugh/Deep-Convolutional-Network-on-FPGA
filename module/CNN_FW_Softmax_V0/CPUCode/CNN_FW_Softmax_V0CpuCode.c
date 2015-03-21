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
	const int dim_1 = 500;
	const int dim_2 = 384;
	const int vec_size = 1;
	const int dim_2_vec = dim_2/vec_size;
	const int dim_3 = 10;

	const int ticks = dim_1*dim_2_vec;
	const int x_offset = 0;
	const int x_size = dim_1*dim_2*sizeof(real);
	const int w_size = dim_1*dim_3*sizeof(real);
	const int sm_offset = x_size;
	const int sm_size = dim_2*dim_3*sizeof(real);
	const int pred_size = dim_2*sizeof(cateType);

	real* x = (real*)malloc(x_size);
	real* w = (real*)malloc(w_size);
	real* sm = (real*)malloc(sm_size);
	cateType* pred = (cateType*)malloc(pred_size);
	
	max_file_t *maxfile = CNN_FW_Softmax_V0_init();
	max_engine_t *engine = max_load(maxfile, "*");

	printf("Writing to LMem.\n");
	max_actions_t* act = max_actions_init(maxfile, "writeLMem");
	max_set_param_uint64t(act, "offset", x_offset);
	max_set_param_uint64t(act, "size", x_size);
	max_queue_input(act, "cpu_to_lmem_at_cpu", x, x_size);
	max_run(engine, act);

	printf("Running on DFE.\n");
	act = max_actions_init(maxfile, "default");
	max_set_param_uint64t(act, "ticks", ticks);
	max_set_param_uint64t(act, "dim_1", dim_1);
	max_set_param_uint64t(act, "x_offset", x_offset);
	max_set_param_uint64t(act, "x_size", x_size);
	max_set_param_uint64t(act, "w_size", w_size);
	max_queue_input(act, "w", w, w_size);
	max_set_param_uint64t(act, "softmax_offset", sm_offset);
	max_set_param_uint64t(act, "softmax_size", sm_size);
	max_set_param_uint64t(act, "pred_size", pred_size);
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
	free(sm);
	free(pred);

	return 0;
}
