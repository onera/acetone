#include <stdio.h>
#include <math.h>
#include "inference.h"
{{#target_specific}}
#include "target.h"
{{/target_specific}}
{{#time}}
#include <time.h>
#include "MMU.h"
#include "PMH.h"

unsigned long long int start, end, difference;

#define counter_id 0x1
#define event_id 0x11
{{/time}}

int inference({{data_type}} prediction[{{output_size}}], {{data_type}} nn_input[{{input_size}}]){
    {{#debug_file}}
    FILE *fp = fopen("{{debug_file}}", "w+");

    {{/debug_file}}
    {{#time}}
    difference = 0;
    select_evt_counter(counter_id);
    event_track(counter_id);
    enable_evt_counter(counter_id);

    {{/time}}
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
    int col_index;
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
    int indice_Gather_{{idx}}[{{length}}] = {{list}};
    {{/indices}}
{{/is_gather}}
{{#is_reduced}}
    {{data_type}} reduced;
{{/is_reduced}}

{{{pre_processing}}}
{{#layers}}
{{#time}}
    reset_evt_counter(0);
    start = read_evt_counter();
{{/time}}
{{{inference_function}}}
{{#time}}
    end = read_evt_counter();
    difference = difference + end - start;
    printf("time {{name}} {{idx}}: %llu\n",end-start);
{{/time}}
{{#debug_layer}}
    fprintf(fp, "{{name}} {{idx}} {{to_transpose}} {{channels}} {{height}} {{width}}\n");
    for (k = 0; k < {{size}}; ++k)
    {
        fprintf(fp,"%.9g ", output_{{path}}[k]);
        if (k == {{size}} - 1)
        {
            fprintf(fp, "\n");
        }
    }
{{/debug_layer}}

    {{#cst}}
    for (k = 0; k < {{size}}; k++)
    {
        cst_{{cst_name}}[k] = output_{{path}}[k];
    }
    {{/cst}}
{{/layers}}

{{#time}}
    reset_evt_counter(0);
    start = read_evt_counter();
{{/time}}
{{{output_str}}}
{{#time}}
    end = read_evt_counter();
    difference = difference + end - start;
    printf("time output: %llu\n",end-start);
    printf("\ntime total time: %llu\n",difference);
{{/time}}


{{{post_processing}}}
    return 0;
}