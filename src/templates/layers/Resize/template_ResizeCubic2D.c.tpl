    // {{name}}_{{idx}}{{comment}}
    a = {{cubic_coeff_a}};

    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode_x}}}
                x0 = floor(x);
                {{{coordinate_transformation_mode_y}}}
                y0 = floor(y);
                
                f_1 = {{output_str}}[y0-1 + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = {{output_str}}[y0-1 + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = {{output_str}}[y0-1 + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = {{output_str}}[y0-1 + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b_1 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                f_1 = {{output_str}}[y0 + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = {{output_str}}[y0 + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = {{output_str}}[y0 + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = {{output_str}}[y0 + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b0 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                f_1 = {{output_str}}[y0+1 + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = {{output_str}}[y0+1 + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = {{output_str}}[y0+1 + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = {{output_str}}[y0+1 + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b1 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                f_1 = {{output_str}}[y0+2 + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = {{output_str}}[y0+2 + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = {{output_str}}[y0+2 + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = {{output_str}}[y0+2 + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b2 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                s = y - y0;

                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = b_1*a*s*(1 + s*(s - 2)) + b0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + b1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + b2*a*s*s*(1 - s);
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