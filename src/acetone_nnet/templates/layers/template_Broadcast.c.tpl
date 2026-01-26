    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                {{#max}}
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = ctx->{{output_str0}}[(j%{{input_width0}}) + {{input_width0}}*((i%{{input_height0}}) + {{input_height0}}*(f % {{input_channels0}}))];
                {{#broadcast}}
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = fmax(ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)],ctx->{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]);
                {{/broadcast}}
                {{/max}}
                {{#min}}
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = ctx->{{output_str0}}[(j%{{input_width0}}) + {{input_width0}}*((i%{{input_height0}}) + {{input_height0}}*(f % {{input_channels0}}))];
                {{#broadcast}}
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = fmin(ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)],ctx->{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]);
                {{/broadcast}}
                {{/min}}
                {{#Average}}
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = ({{#broadcast}}ctx->{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]{{operator}}{{/broadcast}})/{{prev_size}};
                {{/Average}}
                {{#other}}
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{#broadcast}}ctx->{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]{{operator}}{{/broadcast}}{{#constant}}{{operator}}constant_{{name}}_{{idx}}[(j%{{cst_width}}) + {{cst_width}}*((i%{{cst_height}}) + {{cst_height}}*(f % {{cst_channels}}))]{{/constant}};
                {{/other}}
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = {{#activation}}ctx->tensor_temp[k]{{/activation}};
    }