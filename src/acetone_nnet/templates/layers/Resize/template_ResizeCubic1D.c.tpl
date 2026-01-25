    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    a = {{cubic_coeff_a}};

    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for(j = 0; j < {{output_width}}; ++j)
            {
                {{{coordinate_transformation_mode}}}
                x0 = floor(x);

                f_1 = x0-1 < 0 ? ctx->{{output_str}}[0 + {{dimension}}*f] : x0-1 >= {{dimension}} ? ctx->{{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : ctx->{{output_str}}[x0-1 + {{dimension}}*f];
                f0 = x0 < 0 ? ctx->{{output_str}}[0 + {{dimension}}*f] : x0 >= {{dimension}} ? ctx->{{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : ctx->{{output_str}}[x0 + {{dimension}}*f];
                f1 = x0+1 < 0 ? ctx->{{output_str}}[0 + {{dimension}}*f] : x0+1 >= {{dimension}} ? ctx->{{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : ctx->{{output_str}}[x0+1 + {{dimension}}*f];
                f2 = x0+ 2 < 0 ? ctx->{{output_str}}[0 + {{dimension}}*f] : x0+2 >= {{dimension}} ? ctx->{{output_str}}[{{dimension}} - 1 + {{dimension}}*f] : ctx->{{output_str}}[x0+2 + {{dimension}}*f];

                s = x - x0;

                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
            }
        }
    }

    for(k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = ctx->tensor_temp[k];
    }