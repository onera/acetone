#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference(float prediction[1000], float nn_input[150528])
{
    float sum;
    float max;
    int count;

    // Input_layer_0
    for (int i = 0; i < 150528; ++i) 
    { 
        output_0[i] = nn_input[i]; 
    } 

    // Conv2D_1
    for (int k = 0; k < 150528; ++k){
        tensor_temp[k] = output_0[k];
    }
    // im2col
    for (int i = 0; i < 27; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 111; ++h) {
            for (int w = 0; w < 111; ++w) {

                int ii = h * 2 - 0 + i_offset; 
                int jj = w * 2 - 0 + j_offset;

                int j = h*111 + w;
                if (ii >= 0 && ii < 224 && jj >= 0 && jj < 224)
                    output_0[i*12321 + j] = tensor_temp[(c_offset*224 + ii)*224 + jj];
                else
                    output_0[i*12321 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 788544; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<27; ++p){
           register float weight = weights_Conv2D_01[i*27+p];
           for(int j=0; j<12321; ++j){
               tensor_temp[i*12321 + j] += weight * output_0[p*12321 + j];
           }
       }
        for(int j=0; j<12321; ++j){
            register float output = tensor_temp[i*12321 + j];
            output += biases_Conv2D_01[i];
            tensor_temp[i*12321 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 788544; ++k){
        output_0[k] = tensor_temp[k];
    }

    // MaxPooling2D_2
    for (int c = 0; c < 64; ++c)
    {
        for (int i = 0; i < 55; ++i)
        {
            for (int j = 0; j < 55; ++j)
            {
                max = -INFINITY;
                for (int m = 0; m < 3; ++m)
                {
                    for (int n = 0; n < 3; ++n)
                    {
                        int ii = i*2 + m - 0;
                        int jj = j*2 + n - 0;

                        if (ii >= 0 && ii < 111 && jj >= 0 && jj < 111)
                        {
                            if (output_0[jj + 111*(ii + 111*c)] > max)
                                max = output_0[jj + 111*(ii + 111*c)];
                        }
                    }
                }
                output_0[j + 55*(i + 55*c)] = max;

            }
        }
    }

    // Conv2D_3
    for (int k = 0; k < 193600; ++k){
        tensor_temp[k] = output_0[k];
    }
    // im2col
    for (int i = 0; i < 64; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 55; ++h) {
            for (int w = 0; w < 55; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*55 + w;
                if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                    output_0[i*3025 + j] = tensor_temp[(c_offset*55 + ii)*55 + jj];
                else
                    output_0[i*3025 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 48400; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<16; i++){
       for (int p=0; p<64; ++p){
           register float weight = weights_Conv2D_03[i*64+p];
           for(int j=0; j<3025; ++j){
               tensor_temp[i*3025 + j] += weight * output_0[p*3025 + j];
           }
       }
        for(int j=0; j<3025; ++j){
            register float output = tensor_temp[i*3025 + j];
            output += biases_Conv2D_03[i];
            tensor_temp[i*3025 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 48400; ++k){
        output_0[k] = tensor_temp[k];
    }

    for (int k = 0; k < 48400; ++k)
    {
        cst_0[k] = output_0[k];
    }

    // Conv2D_5
    // im2col
    for (int i = 0; i < 144; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 55; ++h) {
            for (int w = 0; w < 55; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*55 + w;
                if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                    output_1[i*3025 + j] = cst_0[(c_offset*55 + ii)*55 + jj];
                else
                    output_1[i*3025 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 193600; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<144; ++p){
           register float weight = weights_Conv2D_05[i*144+p];
           for(int j=0; j<3025; ++j){
               tensor_temp[i*3025 + j] += weight * output_1[p*3025 + j];
           }
       }
        for(int j=0; j<3025; ++j){
            register float output = tensor_temp[i*3025 + j];
            output += biases_Conv2D_05[i];
            tensor_temp[i*3025 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 193600; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Conv2D_4
    // im2col
    for (int i = 0; i < 16; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 55; ++h) {
            for (int w = 0; w < 55; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*55 + w;
                if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                    output_0[i*3025 + j] = cst_0[(c_offset*55 + ii)*55 + jj];
                else
                    output_0[i*3025 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 193600; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<16; ++p){
           register float weight = weights_Conv2D_04[i*16+p];
           for(int j=0; j<3025; ++j){
               tensor_temp[i*3025 + j] += weight * output_0[p*3025 + j];
           }
       }
        for(int j=0; j<3025; ++j){
            register float output = tensor_temp[i*3025 + j];
            output += biases_Conv2D_04[i];
            tensor_temp[i*3025 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 193600; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Concatenate_6
    for (int f = 0; f < 128; f++)
    {
        for (int i = 0; i < 55; i++)
        {
            for (int j = 0; j < 55; j++)
            {
                if((f < 64) && (f >= 0))
                {
                    tensor_temp[j + 55 * (i + 55 * f)] = output_0[j  + 55 * (i + 55 * (f - 0) )];
                }
                if((f < 128) && (f >= 64))
                {
                    tensor_temp[j + 55 * (i + 55 * f)] = output_1[j  + 55 * (i + 55 * (f - 64) )];
                }
            }
        }
    }

    for (int k = 0; k < 387200; ++k){
        output_1[k] = tensor_temp[k];
    }
    // Conv2D_7
    for (int k = 0; k < 387200; ++k){
        tensor_temp[k] = output_1[k];
    }
    // im2col
    for (int i = 0; i < 128; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 55; ++h) {
            for (int w = 0; w < 55; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*55 + w;
                if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                    output_1[i*3025 + j] = tensor_temp[(c_offset*55 + ii)*55 + jj];
                else
                    output_1[i*3025 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 48400; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<16; i++){
       for (int p=0; p<128; ++p){
           register float weight = weights_Conv2D_07[i*128+p];
           for(int j=0; j<3025; ++j){
               tensor_temp[i*3025 + j] += weight * output_1[p*3025 + j];
           }
       }
        for(int j=0; j<3025; ++j){
            register float output = tensor_temp[i*3025 + j];
            output += biases_Conv2D_07[i];
            tensor_temp[i*3025 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 48400; ++k){
        output_1[k] = tensor_temp[k];
    }

    for (int k = 0; k < 48400; ++k)
    {
        cst_0[k] = output_1[k];
    }

    // Conv2D_9
    // im2col
    for (int i = 0; i < 144; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 55; ++h) {
            for (int w = 0; w < 55; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*55 + w;
                if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                    output_0[i*3025 + j] = cst_0[(c_offset*55 + ii)*55 + jj];
                else
                    output_0[i*3025 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 193600; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<144; ++p){
           register float weight = weights_Conv2D_09[i*144+p];
           for(int j=0; j<3025; ++j){
               tensor_temp[i*3025 + j] += weight * output_0[p*3025 + j];
           }
       }
        for(int j=0; j<3025; ++j){
            register float output = tensor_temp[i*3025 + j];
            output += biases_Conv2D_09[i];
            tensor_temp[i*3025 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 193600; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Conv2D_8
    // im2col
    for (int i = 0; i < 16; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 55; ++h) {
            for (int w = 0; w < 55; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*55 + w;
                if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                    output_1[i*3025 + j] = cst_0[(c_offset*55 + ii)*55 + jj];
                else
                    output_1[i*3025 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 193600; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<16; ++p){
           register float weight = weights_Conv2D_08[i*16+p];
           for(int j=0; j<3025; ++j){
               tensor_temp[i*3025 + j] += weight * output_1[p*3025 + j];
           }
       }
        for(int j=0; j<3025; ++j){
            register float output = tensor_temp[i*3025 + j];
            output += biases_Conv2D_08[i];
            tensor_temp[i*3025 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 193600; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Concatenate_10
    for (int f = 0; f < 128; f++)
    {
        for (int i = 0; i < 55; i++)
        {
            for (int j = 0; j < 55; j++)
            {
                if((f < 64) && (f >= 0))
                {
                    tensor_temp[j + 55 * (i + 55 * f)] = output_1[j  + 55 * (i + 55 * (f - 0) )];
                }
                if((f < 128) && (f >= 64))
                {
                    tensor_temp[j + 55 * (i + 55 * f)] = output_0[j  + 55 * (i + 55 * (f - 64) )];
                }
            }
        }
    }

    for (int k = 0; k < 387200; ++k){
        output_0[k] = tensor_temp[k];
    }
    // MaxPooling2D_11
    for (int c = 0; c < 128; ++c)
    {
        for (int i = 0; i < 27; ++i)
        {
            for (int j = 0; j < 27; ++j)
            {
                max = -INFINITY;
                for (int m = 0; m < 3; ++m)
                {
                    for (int n = 0; n < 3; ++n)
                    {
                        int ii = i*2 + m - 0;
                        int jj = j*2 + n - 0;

                        if (ii >= 0 && ii < 55 && jj >= 0 && jj < 55)
                        {
                            if (output_0[jj + 55*(ii + 55*c)] > max)
                                max = output_0[jj + 55*(ii + 55*c)];
                        }
                    }
                }
                output_0[j + 27*(i + 27*c)] = max;

            }
        }
    }

    // Conv2D_12
    for (int k = 0; k < 93312; ++k){
        tensor_temp[k] = output_0[k];
    }
    // im2col
    for (int i = 0; i < 128; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*27 + w;
                if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                    output_0[i*729 + j] = tensor_temp[(c_offset*27 + ii)*27 + jj];
                else
                    output_0[i*729 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 23328; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<32; i++){
       for (int p=0; p<128; ++p){
           register float weight = weights_Conv2D_12[i*128+p];
           for(int j=0; j<729; ++j){
               tensor_temp[i*729 + j] += weight * output_0[p*729 + j];
           }
       }
        for(int j=0; j<729; ++j){
            register float output = tensor_temp[i*729 + j];
            output += biases_Conv2D_12[i];
            tensor_temp[i*729 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 23328; ++k){
        output_0[k] = tensor_temp[k];
    }

    for (int k = 0; k < 23328; ++k)
    {
        cst_0[k] = output_0[k];
    }

    // Conv2D_14
    // im2col
    for (int i = 0; i < 288; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*27 + w;
                if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                    output_1[i*729 + j] = cst_0[(c_offset*27 + ii)*27 + jj];
                else
                    output_1[i*729 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 93312; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<128; i++){
       for (int p=0; p<288; ++p){
           register float weight = weights_Conv2D_14[i*288+p];
           for(int j=0; j<729; ++j){
               tensor_temp[i*729 + j] += weight * output_1[p*729 + j];
           }
       }
        for(int j=0; j<729; ++j){
            register float output = tensor_temp[i*729 + j];
            output += biases_Conv2D_14[i];
            tensor_temp[i*729 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 93312; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Conv2D_13
    // im2col
    for (int i = 0; i < 32; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*27 + w;
                if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                    output_0[i*729 + j] = cst_0[(c_offset*27 + ii)*27 + jj];
                else
                    output_0[i*729 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 93312; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<128; i++){
       for (int p=0; p<32; ++p){
           register float weight = weights_Conv2D_13[i*32+p];
           for(int j=0; j<729; ++j){
               tensor_temp[i*729 + j] += weight * output_0[p*729 + j];
           }
       }
        for(int j=0; j<729; ++j){
            register float output = tensor_temp[i*729 + j];
            output += biases_Conv2D_13[i];
            tensor_temp[i*729 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 93312; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Concatenate_15
    for (int f = 0; f < 256; f++)
    {
        for (int i = 0; i < 27; i++)
        {
            for (int j = 0; j < 27; j++)
            {
                if((f < 128) && (f >= 0))
                {
                    tensor_temp[j + 27 * (i + 27 * f)] = output_0[j  + 27 * (i + 27 * (f - 0) )];
                }
                if((f < 256) && (f >= 128))
                {
                    tensor_temp[j + 27 * (i + 27 * f)] = output_1[j  + 27 * (i + 27 * (f - 128) )];
                }
            }
        }
    }

    for (int k = 0; k < 186624; ++k){
        output_1[k] = tensor_temp[k];
    }
    // Conv2D_16
    for (int k = 0; k < 186624; ++k){
        tensor_temp[k] = output_1[k];
    }
    // im2col
    for (int i = 0; i < 256; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*27 + w;
                if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                    output_1[i*729 + j] = tensor_temp[(c_offset*27 + ii)*27 + jj];
                else
                    output_1[i*729 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 23328; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<32; i++){
       for (int p=0; p<256; ++p){
           register float weight = weights_Conv2D_16[i*256+p];
           for(int j=0; j<729; ++j){
               tensor_temp[i*729 + j] += weight * output_1[p*729 + j];
           }
       }
        for(int j=0; j<729; ++j){
            register float output = tensor_temp[i*729 + j];
            output += biases_Conv2D_16[i];
            tensor_temp[i*729 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 23328; ++k){
        output_1[k] = tensor_temp[k];
    }

    for (int k = 0; k < 23328; ++k)
    {
        cst_0[k] = output_1[k];
    }

    // Conv2D_18
    // im2col
    for (int i = 0; i < 288; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*27 + w;
                if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                    output_0[i*729 + j] = cst_0[(c_offset*27 + ii)*27 + jj];
                else
                    output_0[i*729 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 93312; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<128; i++){
       for (int p=0; p<288; ++p){
           register float weight = weights_Conv2D_18[i*288+p];
           for(int j=0; j<729; ++j){
               tensor_temp[i*729 + j] += weight * output_0[p*729 + j];
           }
       }
        for(int j=0; j<729; ++j){
            register float output = tensor_temp[i*729 + j];
            output += biases_Conv2D_18[i];
            tensor_temp[i*729 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 93312; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Conv2D_17
    // im2col
    for (int i = 0; i < 32; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*27 + w;
                if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                    output_1[i*729 + j] = cst_0[(c_offset*27 + ii)*27 + jj];
                else
                    output_1[i*729 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 93312; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<128; i++){
       for (int p=0; p<32; ++p){
           register float weight = weights_Conv2D_17[i*32+p];
           for(int j=0; j<729; ++j){
               tensor_temp[i*729 + j] += weight * output_1[p*729 + j];
           }
       }
        for(int j=0; j<729; ++j){
            register float output = tensor_temp[i*729 + j];
            output += biases_Conv2D_17[i];
            tensor_temp[i*729 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 93312; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Concatenate_19
    for (int f = 0; f < 256; f++)
    {
        for (int i = 0; i < 27; i++)
        {
            for (int j = 0; j < 27; j++)
            {
                if((f < 128) && (f >= 0))
                {
                    tensor_temp[j + 27 * (i + 27 * f)] = output_1[j  + 27 * (i + 27 * (f - 0) )];
                }
                if((f < 256) && (f >= 128))
                {
                    tensor_temp[j + 27 * (i + 27 * f)] = output_0[j  + 27 * (i + 27 * (f - 128) )];
                }
            }
        }
    }

    for (int k = 0; k < 186624; ++k){
        output_0[k] = tensor_temp[k];
    }
    // MaxPooling2D_20
    for (int c = 0; c < 256; ++c)
    {
        for (int i = 0; i < 13; ++i)
        {
            for (int j = 0; j < 13; ++j)
            {
                max = -INFINITY;
                for (int m = 0; m < 3; ++m)
                {
                    for (int n = 0; n < 3; ++n)
                    {
                        int ii = i*2 + m - 0;
                        int jj = j*2 + n - 0;

                        if (ii >= 0 && ii < 27 && jj >= 0 && jj < 27)
                        {
                            if (output_0[jj + 27*(ii + 27*c)] > max)
                                max = output_0[jj + 27*(ii + 27*c)];
                        }
                    }
                }
                output_0[j + 13*(i + 13*c)] = max;

            }
        }
    }

    // Conv2D_21
    for (int k = 0; k < 43264; ++k){
        tensor_temp[k] = output_0[k];
    }
    // im2col
    for (int i = 0; i < 256; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = tensor_temp[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 8112; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<48; i++){
       for (int p=0; p<256; ++p){
           register float weight = weights_Conv2D_21[i*256+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_21[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 8112; ++k){
        output_0[k] = tensor_temp[k];
    }

    for (int k = 0; k < 8112; ++k)
    {
        cst_0[k] = output_0[k];
    }

    // Conv2D_23
    // im2col
    for (int i = 0; i < 432; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_1[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_1[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 32448; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<192; i++){
       for (int p=0; p<432; ++p){
           register float weight = weights_Conv2D_23[i*432+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_1[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_23[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 32448; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Conv2D_22
    // im2col
    for (int i = 0; i < 48; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 32448; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<192; i++){
       for (int p=0; p<48; ++p){
           register float weight = weights_Conv2D_22[i*48+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_22[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 32448; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Concatenate_24
    for (int f = 0; f < 384; f++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int j = 0; j < 13; j++)
            {
                if((f < 192) && (f >= 0))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_0[j  + 13 * (i + 13 * (f - 0) )];
                }
                if((f < 384) && (f >= 192))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_1[j  + 13 * (i + 13 * (f - 192) )];
                }
            }
        }
    }

    for (int k = 0; k < 64896; ++k){
        output_1[k] = tensor_temp[k];
    }
    // Conv2D_25
    for (int k = 0; k < 64896; ++k){
        tensor_temp[k] = output_1[k];
    }
    // im2col
    for (int i = 0; i < 384; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_1[i*169 + j] = tensor_temp[(c_offset*13 + ii)*13 + jj];
                else
                    output_1[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 8112; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<48; i++){
       for (int p=0; p<384; ++p){
           register float weight = weights_Conv2D_25[i*384+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_1[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_25[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 8112; ++k){
        output_1[k] = tensor_temp[k];
    }

    for (int k = 0; k < 8112; ++k)
    {
        cst_0[k] = output_1[k];
    }

    // Conv2D_27
    // im2col
    for (int i = 0; i < 432; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 32448; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<192; i++){
       for (int p=0; p<432; ++p){
           register float weight = weights_Conv2D_27[i*432+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_27[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 32448; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Conv2D_26
    // im2col
    for (int i = 0; i < 48; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_1[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_1[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 32448; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<192; i++){
       for (int p=0; p<48; ++p){
           register float weight = weights_Conv2D_26[i*48+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_1[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_26[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 32448; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Concatenate_28
    for (int f = 0; f < 384; f++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int j = 0; j < 13; j++)
            {
                if((f < 192) && (f >= 0))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_1[j  + 13 * (i + 13 * (f - 0) )];
                }
                if((f < 384) && (f >= 192))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_0[j  + 13 * (i + 13 * (f - 192) )];
                }
            }
        }
    }

    for (int k = 0; k < 64896; ++k){
        output_0[k] = tensor_temp[k];
    }
    // Conv2D_29
    for (int k = 0; k < 64896; ++k){
        tensor_temp[k] = output_0[k];
    }
    // im2col
    for (int i = 0; i < 384; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = tensor_temp[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 10816; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<384; ++p){
           register float weight = weights_Conv2D_29[i*384+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_29[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 10816; ++k){
        output_0[k] = tensor_temp[k];
    }

    for (int k = 0; k < 10816; ++k)
    {
        cst_0[k] = output_0[k];
    }

    // Conv2D_31
    // im2col
    for (int i = 0; i < 576; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_1[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_1[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 43264; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<256; i++){
       for (int p=0; p<576; ++p){
           register float weight = weights_Conv2D_31[i*576+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_1[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_31[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 43264; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Conv2D_30
    // im2col
    for (int i = 0; i < 64; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 43264; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<256; i++){
       for (int p=0; p<64; ++p){
           register float weight = weights_Conv2D_30[i*64+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_30[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 43264; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Concatenate_32
    for (int f = 0; f < 512; f++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int j = 0; j < 13; j++)
            {
                if((f < 256) && (f >= 0))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_0[j  + 13 * (i + 13 * (f - 0) )];
                }
                if((f < 512) && (f >= 256))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_1[j  + 13 * (i + 13 * (f - 256) )];
                }
            }
        }
    }

    for (int k = 0; k < 86528; ++k){
        output_1[k] = tensor_temp[k];
    }
    // Conv2D_33
    for (int k = 0; k < 86528; ++k){
        tensor_temp[k] = output_1[k];
    }
    // im2col
    for (int i = 0; i < 512; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_1[i*169 + j] = tensor_temp[(c_offset*13 + ii)*13 + jj];
                else
                    output_1[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 10816; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<64; i++){
       for (int p=0; p<512; ++p){
           register float weight = weights_Conv2D_33[i*512+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_1[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_33[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 10816; ++k){
        output_1[k] = tensor_temp[k];
    }

    for (int k = 0; k < 10816; ++k)
    {
        cst_0[k] = output_1[k];
    }

    // Conv2D_35
    // im2col
    for (int i = 0; i < 576; ++i) {

        int i_offset = (i / 3) % 3;
        int j_offset = i % 3;
        int c_offset = i / 3 / 3;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 1 + i_offset; 
                int jj = w * 1 - 1 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 43264; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<256; i++){
       for (int p=0; p<576; ++p){
           register float weight = weights_Conv2D_35[i*576+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_35[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 43264; ++k){
        output_0[k] = tensor_temp[k];
    }

    // Conv2D_34
    // im2col
    for (int i = 0; i < 64; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_1[i*169 + j] = cst_0[(c_offset*13 + ii)*13 + jj];
                else
                    output_1[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 43264; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<256; i++){
       for (int p=0; p<64; ++p){
           register float weight = weights_Conv2D_34[i*64+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_1[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_34[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 43264; ++k){
        output_1[k] = tensor_temp[k];
    }

    // Concatenate_36
    for (int f = 0; f < 512; f++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int j = 0; j < 13; j++)
            {
                if((f < 256) && (f >= 0))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_1[j  + 13 * (i + 13 * (f - 0) )];
                }
                if((f < 512) && (f >= 256))
                {
                    tensor_temp[j + 13 * (i + 13 * f)] = output_0[j  + 13 * (i + 13 * (f - 256) )];
                }
            }
        }
    }

    for (int k = 0; k < 86528; ++k){
        output_0[k] = tensor_temp[k];
    }
    // Conv2D_37
    for (int k = 0; k < 86528; ++k){
        tensor_temp[k] = output_0[k];
    }
    // im2col
    for (int i = 0; i < 512; ++i) {

        int i_offset = (i / 1) % 1;
        int j_offset = i % 1;
        int c_offset = i / 1 / 1;

        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {

                int ii = h * 1 - 0 + i_offset; 
                int jj = w * 1 - 0 + j_offset;

                int j = h*13 + w;
                if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                    output_0[i*169 + j] = tensor_temp[(c_offset*13 + ii)*13 + jj];
                else
                    output_0[i*169 + j] = 0;
            }
        }
    }
    

    for (int k = 0; k < 169000; ++k){
        tensor_temp[k] = 0;
    }
    // gemm_nn
    for (int i=0; i<1000; i++){
       for (int p=0; p<512; ++p){
           register float weight = weights_Conv2D_37[i*512+p];
           for(int j=0; j<169; ++j){
               tensor_temp[i*169 + j] += weight * output_0[p*169 + j];
           }
       }
        for(int j=0; j<169; ++j){
            register float output = tensor_temp[i*169 + j];
            output += biases_Conv2D_37[i];
            tensor_temp[i*169 + j] = output > 0 ? output : 0;
        }
    }

    for (int k = 0; k < 169000; ++k){
        output_0[k] = tensor_temp[k];
    }

    // AveragePooling2D_38
    for (int c = 0; c < 1000; ++c)
    {
        for (int i = 0; i < 1; ++i)
        {
            for (int j = 0; j < 1; ++j)
            {
                sum = 0; count = 0;
                for (int m = 0; m < 13; ++m)
                {
                    for (int n = 0; n < 13; ++n)
                    {
                        int ii = i*0 + m - 0;
                        int jj = j*0 + n - 0;

                        if (ii >= 0 && ii < 13 && jj >= 0 && jj < 13)
                        {
                            sum += output_0[jj + 13*(ii + 13*c)];
                            count ++;
                        }
                    }
                }
                output_0[j + 1*(i + 1*c)] = sum/count;

            }
        }
    }

    // Softmax_39
    sum = 0;

    for (int i = 0; i < 1000; ++i)
        sum += exp(output_0[i]);

    for (int j = 0; j < 1000; ++j)
        output_0[j] = exp(output_0[j])/sum;

    for (int k = 0; k < 1000; ++k)
        prediction[k] = output_0[k];

    return 0;
}