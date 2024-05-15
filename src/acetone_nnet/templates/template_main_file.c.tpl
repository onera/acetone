#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include "test_dataset.h" 
#include "inference.h"
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

    char *path = argv[1];

    FILE *fp = fopen(path, "w+");

    {{data_type}} predictions[nb_samples][nn_output_size];

    clock_t t0 = clock();
    for (i = 0; i < nb_samples; ++i)
    {
        inference(predictions[i], nn_test_inputs[i]);
    }
    clock_t t1 = clock();

    printf("   Average time over %d tests: %e s \n", nb_samples,
        (float)(t1-t0)/(float)CLOCKS_PER_SEC/(float)100);

    printf("   ACETONE framework's inference output: \n");
    for (i = 0; i < nb_samples; ++i)
    {
        for (j = 0; j < nn_output_size; ++j)
        {
            fprintf(fp,"%.9g ", predictions[i][j]);
            printf("%.9g ", predictions[i][j]);
            if (j == nn_output_size - 1)
            {
                fprintf(fp, "\n");
                printf("\n");
            }
        }
    }

    fclose(fp);
    fp = NULL;

    return 0;
}