    // {{name}}_{{idx}}{{comment}}
    for (int c = 0; c < {{input_channels}}; ++c)
    {
        for (int i = 0; i < {{output_height}}; ++i)
        {
            for (int j = 0; j < {{output_width}}; ++j)
            {
                {{{update_local_vars}}}
                for (int m = 0; m < {{pool_size}}; ++m)
                {
                    for (int n = 0; n < {{pool_size}}; ++n)
                    {
                        int ii = i*{{strides}} + m - {{pad_left}};
                        int jj = j*{{strides}} + n - {{pad_top}};
                        if (ii >= 0 && ii < {{input_height}} && jj >= 0 && jj < {{input_width}})
                        {
                            {{{specific_function}}}
                        }
                    }
                }
                {{^fused_layer}}
                output_{{road}}[j + {{output_width}}*(i + {{output_height}}*c)] = {{{activation_function}}};
                {{/fused_layer}}
                {{#fused_layer}}
                    {{^linear}}
                {{local_var}} = {{{activation_function}}};
                    {{/linear}}
                output_{{road}}[j + {{output_width}}*(i + {{output_height}}*c)] = {{{fused_layer}}};
                {{/fused_layer}}
            }
        }
    }