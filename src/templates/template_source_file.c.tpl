#include <stdio.h>
#include <math.h>
#include "inference.h"

int inference({{data_type}} prediction[{{output_size}}], {{data_type}} nn_input[{{input_size}}]){
    int f;
    int i;
    int j;
    int k;
    {{#p}}
    int p;
    {{/p}}
    {{#hw}}
    int h;
    int w;
    {{/hw}}

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
    int x0;
    int y0;
{{/is_resize}}
{{#is_cubic_interpolation}}
    float a;
    float f_1;
    float f0;
    float f1;
    float f2;
    float s;
{{/is_cubic_interpolation}}
{{#is_linear_interpolation}}
    int y1;
    int x1;
    float f11;
    float f12;
    float f21;
    float f22;
{{/is_linear_interpolation}}
{{#is_gather}}
    int position;
    {{#indices}}
    int indice_Gather_{{idx}}[{{lenght}}] = {{list}};
    {{/indices}}
{{/is_gather}}

{{{pre_processing}}}
{{#layers}}
{{{inference_function}}}

    {{#cst}}
    for (k = 0; k < {{size}}; k++)
    {
        cst_{{cst_name}}[k] = output_{{path}}[k];
    }
    {{/cst}}
{{/layers}}

{{{ouput_str}}}

{{{post_processing}}}
    return 0;
}