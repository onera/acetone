#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "inference.h"
#include "test_dataset.h"

void batch_loop(float * restrict predictions, float * restrict nn_input, int batch_size)
{
    int i,j;
    for (j = 0; j < batch_size; j+=MAX_BATCH_SIZE){
        int fin_bloc = j + MAX_BATCH_SIZE > batch_size ? batch_size % MAX_BATCH_SIZE : MAX_BATCH_SIZE;
        #pragma omp parallel for num_threads(MAX_BATCH_SIZE)
        for (i = 0; i < fin_bloc; ++i)
        {
            inference(&Context[i],&predictions[(j+i)*nn_output_size], &nn_input[(j+i)*nn_input_size]);
        }
    }
}

void copy_weights(
{{memcpy_params}}
){
    {{#nb_params}}
    memcpy({{param_name}},model_{{param_name}},{{param_size}}*sizeof({{dtype}}));
    {{/nb_params}}
}

int save_weights(char *file_path){
    FILE *f;
    int nb_written=0;
    f = fopen(file_path,"wb");
    {{#nb_params}}
    nb_written += fwrite({{param_name}},1,{{param_size}}*sizeof({{dtype}}),f);
    {{/nb_params}}
    fclose(f);
    return nb_written;
}