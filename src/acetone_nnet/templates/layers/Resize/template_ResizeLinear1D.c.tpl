    // {{name}}_{{idx}}{{comment}}
    x1 = {{len_dimension}};
    x0 = 0;

    for (f = 0; f < {{output_channels}}; ++f)
    {
        f11 = {{output_str}}[x0 + {{dimension}}*f];
        f22 = {{output_str}}[x1 + {{dimension}}*f];

        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode}}}
                x = x<x0 ? x0 : (x>x1 ? x1: x);

                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = (f11*(x1 - x) + f22*(x - x0))/(x1 - x0);
                {{#linear}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
                {{/linear}}
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