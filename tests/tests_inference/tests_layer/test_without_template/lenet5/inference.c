#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference(float prediction[10], float nn_input[784])
{
    static float output_pre[3456];
    static float output_cur[3456];
    float dotproduct;
    float sum;
    float max;
    int count;

    // Input_layer_0
    for (int i = 0; i < 784; ++i) 
    { 
        output_pre[i] = nn_input[i]; 
    } 

    // Conv2D_1
    for (int f = 0; f < 6; ++f)
    {
        for (int i = 0; i < 24; ++i)
        {
            for (int j = 0; j < 24; ++j)
            {
                sum = 0;
                for (int c = 0; c < 1; ++c)
                {
                    for (int m = 0; m < 5; ++m)
                    {
                        for (int n = 0; n < 5; ++n)
                        {
                            int ii = i*1 + m*1 - 0;
                            int jj = j*1 + n*1 - 0;

                            if (ii >= 0 && ii < 28 && jj >= 0 && jj < 28)
                            {
                                sum += output_pre[(ii*28 + jj)*1 + c] * weights_Conv2D_01[((m*5 + n)*1 + c)*6 + f];
                            }
                        }
                    }
                }
                sum += biases_Conv2D_01[f];
                output_cur[(i*24 + j)*6 + f] = (exp(sum)-exp(-sum))/(exp(sum)+exp(-sum));
            }
        }
    }

    for (int k = 0; k < 3456; ++k)
        output_pre[k] = output_cur[k];

    // AveragePooling2D_2
    for (int c = 0; c < 6; ++c)
    {
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                sum = 0; count = 0;
                for (int m = 0; m < 2; ++m)
                {
                    for (int n = 0; n < 2; ++n)
                    {
                        int ii = i*2 + m - 0;
                        int jj = j*2 + n - 0;

                        if (ii >= 0 && ii < 24 && jj >= 0 && jj < 24)
                        {
                            sum += output_pre[(ii*24 + jj)*6 + c];
                            count ++;
                        }
                    }
                }
                output_cur[(i*12 + j)*6 + c] = sum/count;

            }
        }
    }

    for (int k = 0; k < 864; ++k)
        output_pre[k] = output_cur[k];

    // Conv2D_3
    for (int f = 0; f < 16; ++f)
    {
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                sum = 0;
                for (int c = 0; c < 6; ++c)
                {
                    for (int m = 0; m < 5; ++m)
                    {
                        for (int n = 0; n < 5; ++n)
                        {
                            int ii = i*1 + m*1 - 0;
                            int jj = j*1 + n*1 - 0;

                            if (ii >= 0 && ii < 12 && jj >= 0 && jj < 12)
                            {
                                sum += output_pre[(ii*12 + jj)*6 + c] * weights_Conv2D_03[((m*5 + n)*6 + c)*16 + f];
                            }
                        }
                    }
                }
                sum += biases_Conv2D_03[f];
                output_cur[(i*8 + j)*16 + f] = (exp(sum)-exp(-sum))/(exp(sum)+exp(-sum));
            }
        }
    }

    for (int k = 0; k < 1024; ++k)
        output_pre[k] = output_cur[k];

    // AveragePooling2D_4
    for (int c = 0; c < 16; ++c)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                sum = 0; count = 0;
                for (int m = 0; m < 2; ++m)
                {
                    for (int n = 0; n < 2; ++n)
                    {
                        int ii = i*2 + m - 0;
                        int jj = j*2 + n - 0;

                        if (ii >= 0 && ii < 8 && jj >= 0 && jj < 8)
                        {
                            sum += output_pre[(ii*8 + jj)*16 + c];
                            count ++;
                        }
                    }
                }
                output_cur[(i*4 + j)*16 + c] = sum/count;

            }
        }
    }

    for (int k = 0; k < 256; ++k)
        output_pre[k] = output_cur[k];

    // Dense_5
    for (int i = 0; i < 120; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 256; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_05[(i + 120*j)];
        }
        dotproduct += biases_Dense_05[i];
        output_cur[i] = (exp(dotproduct)-exp(-dotproduct))/(exp(dotproduct)+exp(-dotproduct));
    }

    for (int k = 0; k < 120; ++k)
        output_pre[k] = output_cur[k];

    // Dense_6
    for (int i = 0; i < 84; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 120; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_06[(i + 84*j)];
        }
        dotproduct += biases_Dense_06[i];
        output_cur[i] = (exp(dotproduct)-exp(-dotproduct))/(exp(dotproduct)+exp(-dotproduct));
    }

    for (int k = 0; k < 84; ++k)
        output_pre[k] = output_cur[k];

    // Dense_7
    for (int i = 0; i < 10; ++i) 
    { 
        dotproduct = 0;
        for (int j = 0; j < 84; ++j)
        {
            dotproduct += output_pre[j] * weights_Dense_07[(i + 10*j)];
        }
        dotproduct += biases_Dense_07[i];
        output_cur[i] = dotproduct;
    }

    for (int k = 0; k < 10; ++k)
        output_pre[k] = output_cur[k];

    // Softmax_8
    sum = 0;

    for (int i = 0; i < 10; ++i)
        sum += exp(output_pre[i]);

    for (int j = 0; j < 10; ++j)
        output_cur[j] = exp(output_pre[j])/sum;

    for (int k = 0; k < 10; ++k)
        prediction[k] = output_cur[k];

    return 0;
}