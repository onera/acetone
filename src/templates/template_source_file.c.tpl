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

{{#layers}}
{{{inference_function}}}

    {{#cst}}
    for (int k; k < {{size}}; k++)
    {
        cst_{{cst_name}}[k] = output_{{road}}[k];
    }

    {{/cst}}
    {{#is_last}}
    for (int k = 0; k < {{size}}; k++)
    {
        prediction[k] = output_{{road}}[k];
    }

    {{/is_last}}    
{{/layers}}
    return 0;
}