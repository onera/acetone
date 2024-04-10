#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference(float prediction[192], float nn_input[300])
{
    float sum;
    int count;

    // Input_layer_0
    for (int i = 0; i < 300; ++i) 
    { 
        output_pre_0[i] = nn_input[i]; 
    } 

    // AveragePooling2D_1
    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                sum = 0; count = 0;
                for (int m = 0; m < 3; ++m)
                {
                    for (int n = 0; n < 3; ++n)
                    {
                        int ii = i*1 + m - 0;
                        int jj = j*1 + n - 0;

                        if (ii >= 0 && ii < 10 && jj >= 0 && jj < 10)
                        {
                            sum += output_pre_0[c + 3*(jj + 10*ii)];
                            count ++;
                        }
                    }
                }
                output_cur_0[j + 8*(i + 8*c)] = sum/count;

            }
        }
    }

    for (int f = 0; f < 3; ++f){
        for (int i = 0; i < 8; ++i){
            for (int j = 0; j < 8; ++j){
                prediction[(i * 8 + j) * 3 + f] = output_cur_0[(f * 8 + i) * 8 + j];

            }
        }
    }

    return 0;
}