#include <string.h>
#include "inference.h"
#include "test_dataset.h"
void batch_loop(float *prediction, float *nn_input, int bath_size)
{
    int i;
    for (i = 0; i < nb_samples; ++i)
    {
        inference(&prediction[i*nn_output_size], &nn_input[i*nn_input_size]);
    }
}


void copy_weights(
{{memcpy_params}}
){
{{memcpy_code}}
}