#include <string.h>
#include <stdio.h>
#include "inference.h"
#include "test_dataset.h"
void batch_loop(float *prediction, float *nn_input, int batch_size)
{
    int i;
    for (i = 0; i < batch_size; ++i)
    {
        inference(&prediction[i*nn_output_size], &nn_input[i*nn_input_size]);
    }
}

void copy_weights(
{{memcpy_params}}
){
    {{#nb_params}}
    memcpy({{param_name}},model_{{param_name}},{{param_size}}*sizeof({{dtype}}));
    {{/nb_params}}
}

int save_weights(file_path){
    FILE *f;
    int nb_written=0;
    f = fopen(file_path,"wb");
    {{#nb_params}}
    nb_written += fwrite({{param_name}},1,{{param_size}}*sizeof({{dtype}}),f);
    {{/nb_params}}
    fclose(f);
    return nb_written;
}