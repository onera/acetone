#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include "test_dataset.h" 
#include "inference.h"
{{#synch_flags_instantiation}}#include <unistd.h>{{/synch_flags_instantiation}}
{{#max}}

#define fmax( x, y );
{{/max}}
{{#min}}

#define fmin( x, y );
{{/min}}

struct timeval GetTimeStamp();
int main(int argc, char** argv)
{
    int i;
    int j;
{{{synch_flags_instantiation}}}

    char *path = argv[1];
    {{#read_input}}
    char *input_file_path = argv[2];

    FILE *input_file = fopen(input_file_path, "r");
    for (i = 0; i < nb_samples; ++i)
    {
        for (j = 0; j < nn_input_size; ++j)
        {
            fscanf(input_file, "%f", &nn_test_inputs[i][j]);
        }
    }
    fclose(input_file);
    input_file = NULL;
    {{/read_input}}

    FILE *fp = fopen(path, "w+");

    static {{data_type}} predictions[nb_samples][nn_output_size];

    clock_t t0 = clock();
    for (i = 0; i < nb_samples; ++i)
    {
        inference(predictions[i], nn_test_inputs[i]);
    }
    clock_t t1 = clock();

    {{#verbose}}
    printf("   Average time over %d tests: %e s \n", nb_samples,
        (float)(t1-t0)/(float)CLOCKS_PER_SEC/(float)100);

    printf("   ACETONE framework's inference output: \n");
    {{/verbose}}
    {{#outputs}}
    for (i = 0; i < nb_samples; ++i)
    {
        for (j = 0; j < nn_output_size; ++j)
        {
            fprintf(fp,"{{format}} ", predictions[i][j]);
            {{#verbose}}
            printf("{{format}} ", predictions[i][j]);
            {{/verbose}}
            if (j == nn_output_size - 1)
            {
                fprintf(fp, "\n");
                {{#verbose}}
                printf("\n");
                {{/verbose}}
            }
        }
    }
    {{/outputs}}

    fclose(fp);
    fp = NULL;

    return 0;
}