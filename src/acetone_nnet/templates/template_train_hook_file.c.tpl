#include <string.h>
#include "inference.h"
#include "test_dataset.h"
void batch_loop(float *prediction[nn_output_size], float *nn_input[nn_input_size], int bath_size)
{
    int i;
    for (i = 0; i < nb_samples; ++i)
    {
        inference(prediction[i], nn_input[i]);
    }
}


void copy_weights(
{{memcpy_params}}
){
{{memcpy_code}}
}