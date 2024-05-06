    // {{name}}_{{idx}}{{comment}}
    a = {{cubic_coeff_a}};

    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode}}}
                x0 = floor(x);

                f_1 = x0-1 < 0 ? {{output_str}}[0 + {{dimension}}*f] : x0-1 >= {{dimension}} ? {{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : {{output_str}}[x0-1 + {{dimension}}*f];
                f0 = x0 < 0 ? {{output_str}}[0 + {{dimension}}*f] : x0 >= {{dimension}} ? {{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : {{output_str}}[x0 + {{dimension}}*f];
                f1 = x0+1 < 0 ? {{output_str}}[0 + {{dimension}}*f] : x0+1 >= {{dimension}} ? {{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : {{output_str}}[x0+1 + {{dimension}}*f];
                f2 = x0+ 2 < 0 ? {{output_str}}[0 + {{dimension}}*f] : x0+2 >= {{dimension}} ? {{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : {{output_str}}[x0+2 + {{dimension}}*f];

                s = x - x0;

                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);
                {{#activation_function}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
                {{/activation_function}}
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