    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = 0;
                for (k = 0; k < {{shared_dimension}}; ++k)
                {
                    tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] += {{output_str_left}}[k + {{shared_dimension}}*(i + {{output_height}}*f)]*{{output_str_right}}[j + {{output_width}}*(k + {{shared_dimension}}*f)];
                }
                {{#non_linear}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
                {{/non_linear}}
                {{#fused_layer}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{fused_layer}}};
                {{/fused_layer}}
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }