#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference(float prediction[5], float nn_input[5])
{
    static float output_pre[128];
    static float output_cur[128];
    float dotproduct;
    float sum;
    float max;
    int count;

    // Input_layer_0
    for (int i = 0; i < 5; ++i) 
    { 
        output_pre[i] = nn_input[i]; 
    } 

    // Dense_1
    for (int i = 0; i < 128; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 5; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_01[(i + 128*j)];
        }
        dotproduct += biases_Dense_01[i];
        output_cur[i] = dotproduct > 0 ? dotproduct : 0;
    }

    for (int k = 0; k < 128; ++k)
        output_pre[k] = output_cur[k];

    // Dense_2
    for (int i = 0; i < 128; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 128; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_02[(i + 128*j)];
        }
        dotproduct += biases_Dense_02[i];
        output_cur[i] = dotproduct > 0 ? dotproduct : 0;
    }

    for (int k = 0; k < 128; ++k)
        output_pre[k] = output_cur[k];

    // Dense_3
    for (int i = 0; i < 64; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 128; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_03[(i + 64*j)];
        }
        dotproduct += biases_Dense_03[i];
        output_cur[i] = dotproduct > 0 ? dotproduct : 0;
    }

    for (int k = 0; k < 64; ++k)
        output_pre[k] = output_cur[k];

    // Dense_4
    for (int i = 0; i < 32; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 64; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_04[(i + 32*j)];
        }
        dotproduct += biases_Dense_04[i];
        output_cur[i] = dotproduct > 0 ? dotproduct : 0;
    }

    for (int k = 0; k < 32; ++k)
        output_pre[k] = output_cur[k];

    // Dense_5
    for (int i = 0; i < 16; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 32; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_05[(i + 16*j)];
        }
        dotproduct += biases_Dense_05[i];
        output_cur[i] = dotproduct > 0 ? dotproduct : 0;
    }

    for (int k = 0; k < 16; ++k)
        output_pre[k] = output_cur[k];

    // Dense_6
    for (int i = 0; i < 5; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 16; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_06[(i + 5*j)];
        }
        dotproduct += biases_Dense_06[i];
        output_cur[i] = dotproduct;
    }

    for (int k = 0; k < 5; ++k)
        prediction[k] = output_cur[k];

    return 0;
}