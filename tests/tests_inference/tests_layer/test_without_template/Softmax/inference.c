#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference(float prediction[75], float nn_input[200])
{
    float dotproduct;
    float sum;
    // Input_layer_0
    for (int i = 0; i < 200; ++i) 
    { 
        output_pre_0[i] = nn_input[i]; 
    } 

    // Dense_1
    for (int i = 0; i < 75; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 200; ++j)
        {
            dotproduct += output_pre_0[j] * weights_Dense_01[(j + 200*i)];
        }
        dotproduct += biases_Dense_01[i];
        output_cur_0[i] = dotproduct;
    }

    for (int k = 0; k < 75; ++k)
        output_pre_0[k] = output_cur_0[k];

    // Softmax_2
    sum = 0;

    for (int i = 0; i < 75; ++i)
        sum += exp(output_pre_0[i]);

    for (int j = 0; j < 75; ++j)
        output_cur_0[j] = exp(output_pre_0[j])/sum;

    for (int k = 0; k < 75; ++k)
        prediction[k] = output_cur_0[k];

    return 0;
}