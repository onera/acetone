    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                if((f >= {{pads_front}}) && (f < {{channels_without_pads_back}}) && (i >= {{pads_top}}) && (i < {{height_without_pads_bottom}}) && (j >= {{pads_left}}) && (j <{{width_without_pads_right}}))
                {
                    tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{output_str}}[(j -{{pads_left}}) + {{input_width}} * ((i - {{pads_top}}) + {{input_height}} * (f - {{pads_front}}))];
                }
                else
                {
                    tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{constant}};
                }
                {{^linear}}
                tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{{activation_function}}};
                {{/linear}}
                {{#fused_layer}}
                tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{{fused_layer}}};
                {{/fused_layer}}
            }
        }
    }
    for(k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }