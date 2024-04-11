    // {{name}}_{{idx}}{{comment}}
    x1 = {{len_height}};
    x0 = 0;
    y0 = 0;
    y1 = {{len_width}};

    for (f = 0; f < {{output_channels}}; ++f)
    {
        f11 = {{output_str}}[y0 + {{input_width}}*(x0 + {{input_height}}*f)];
        f12 = {{output_str}}[y0 + {{input_width}}*(x1 + {{input_height}}*f)];
        f22 = {{output_str}}[y1 + {{input_width}}*(x1 + {{input_height}}*f)];
        f21 = {{output_str}}[y1 + {{input_width}}*(x0 + {{input_height}}*f)];

        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode_x}}}
                {{{coordinate_transformation_mode_y}}}
                x = x<x0 ? x0 : (x>x1 ? x1: x);
                y = y<y0 ? y0 : (y>y1 ? y1: y);

                tensor_temp[i + {{output_height}}*(j + {{output_width}}*f)] = (f11*(x1 - x)*(y1 - y) + f21*(x - x0)*(y1 - y) + f12*(x1 - x)*(y - y0) + f22*(x - x0)*(y - y0))/((x1 - x0)*(y1 - y0));
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