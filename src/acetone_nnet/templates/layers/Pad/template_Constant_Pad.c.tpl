    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                if((f >= {{pads_front}}) && (f < {{channels_without_pads_back}}) && (i >= {{pads_top}}) && (i < {{height_without_pads_bottom}}) && (j >= {{pads_left}}) && (j <{{width_without_pads_right}}))
                {
                    ctx->tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = ctx->{{output_str}}[(j -{{pads_left}}) + {{input_width}} * ((i - {{pads_top}}) + {{input_height}} * (f - {{pads_front}}))];
                }
                else
                {
                    ctx->tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{constant}};
                }
                ctx->tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{{activation_function}}};
            }
        }
    }
    for(k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = ctx->tensor_temp[k];
    }