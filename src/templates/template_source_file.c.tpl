#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference({{data_type}} prediction[{{output_size}}], {{data_type}} nn_input[{{input_size}}]){

{{#is_dense}}
    {{data_type}} dotproduct;
{{/is_dense}}
{{#is_sum}}
    {{data_type}} sum;
{{/is_sum}}
{{#is_max}}
    {{data_type}} max;
{{/is_max}}
{{#is_count}}
    int count;
{{/is_count}}
{{#is_resize}}
    float x;
    float y;
{{/is_resize}}
{{#is_cubic_interpolation}}
    float a;
    float result_interpolation;
    float f_1;
    float f0;
    float f1;
    float f2;
    float s;
{{/is_cubic_interpolation}}
{{#is_linear_interpolation}}
    int y2;
    int y1;
    int x2;
    int x1;
    float f11;
    float f12;
    float f21;
    float f22;
{{/is_linear_interpolation}}

{{{pre_processing}}}
{{#layers}}
{{{inference_function}}}

    {{#cst}}
    for (int k; k < {{size}}; k++)
    {
        cst_{{cst_name}}[k] = output_{{road}}[k];
    }
    {{/cst}}
{{/layers}}

{{^channels_last}}
    for (int k = 0; k < {{output_size}}; k++)
    {
        prediction[k] = output_{{road}}[k];
    }

{{/channels_last}}
{{#channels_last}}
    for(int f = 0; f < {{output_channels}}; ++f)
    {
        for (int i = 0; i < {{output_height}}; ++i)
        {
            for (int j = 0; j < {{output_width}};  ++j)
            {
                prediction[(i*{{output_width}} + j)*{{output_channels}} + f] = output_{{road}}[(f*{{output_height}} + i)*{{output_width}} + j];
            }
        }
    }

{{/channels_last}} 
{{{post_processing}}}
    return 0;
}