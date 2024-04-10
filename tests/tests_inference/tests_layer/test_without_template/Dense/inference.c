#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference(float prediction[250], float nn_input[500])
{
    float dotproduct;
    float sum;
    // Input_layer_0
    for (int i = 0; i < 500; ++i) 
    { 
        output_pre_0[i] = nn_input[i]; 
    } 

    // Dense_1
    for (int i = 0; i < 250; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 500; ++j)
        {
            dotproduct += output_pre_0[j] * weights_Dense_01[(j + 500*i)];
        }
        dotproduct += biases_Dense_01[i];
        output_cur_0[i] = dotproduct;
    }

    for (int k = 0; k < 250; ++k)
        prediction[k] = output_cur_0[k];

    return 0;
}