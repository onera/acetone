    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode_x}}}
                {{{coordinate_transformation_mode_y}}}
                {{nearest_mode_x}}
                {{nearest_mode_y}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
                {{#fused_layer}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{fused_layer}}};
                {{/fused_layer}}
            }
        }
    }

    for(k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }