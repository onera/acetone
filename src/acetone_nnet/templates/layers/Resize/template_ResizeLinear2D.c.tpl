    // {{name}}_{{idx}}{{comment}}

    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode_x}}}
                {{{coordinate_transformation_mode_y}}}
                x = x<0 ? 0 : (x>{{len_height}} ? {{len_height}}: x);
                y = y<0 ? 0 : (y>{{len_width}} ? {{len_width}}: y);

                x1 = floor(x) + 1;
                x0 = floor(x);
                y0 = floor(y);
                y1 = floor(y) + 1;

                f11 = {{output_str}}[y0 + {{input_width}}*(x0 + {{input_height}}*f)];
                f21 = x1 >= {{input_height}} ? 0 : {{output_str}}[y0 + {{input_width}}*(x1 + {{input_height}}*f)];
                f22 = x1 >= {{input_height}} || y1 >= {{input_width}}? 0 : {{output_str}}[y1 + {{input_width}}*(x1 + {{input_height}}*f)];
                f12 = y1 >= {{input_width}} ? 0 : {{output_str}}[y1 + {{input_width}}*(x0 + {{input_height}}*f)];

                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = (f11*(x1 - x)*(y1 - y) + f21*(x - x0)*(y1 - y) + f12*(x1 - x)*(y - y0) + f22*(x - x0)*(y - y0))/((x1 - x0)*(y1 - y0));
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