#include <stdio.h>
#include <math.h>
#include "inference.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) > (b) ? (a) : (b))
int inference(float prediction[300], float nn_input[300])
{
    float sum;
    // Input_layer_0
    for (int i = 0; i < 300; ++i) 
    { 
        output_pre_0[i] = nn_input[i]; 
    } 

    // Conv2D_1
    for (int f = 0; f < 3; ++f)
    {
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                sum = 0;
                for (int c = 0; c < 3; ++c)
                {
                    for (int m = 0; m < 3; ++m)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            int ii = i*1 + m*1 - 1;
                            int jj = j*1 + n*1 - 1;

                            if (ii >= 0 && ii < 10 && jj >= 0 && jj < 10)
                            {
                                sum += output_pre_0[(ii*10 + jj)*3 + c] * weights_Conv2D_01[n + 3*(m + 3*(c + 3*f))];
                            }
                        }
                    }
                }
                sum += biases_Conv2D_01[f];
                output_cur_0[j + 10*(i + 10*f)] = sum;
            }
        }
    }

    for (int f = 0; f < 3; ++f){
        for (int i = 0; i < 10; ++i){
            for (int j = 0; j < 10; ++j){
                prediction[(i * 10 + j) * 3 + f] = output_cur_0[(f * 10 + i) * 10 + j];

            }
        }
    }

    return 0;
}