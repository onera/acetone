    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
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

                col_index = y0-1 < 0 ? 0 : y0-1 >= {{input_width}} ? {{input_width}} - 1 : y0-1;
                f_1 = x0-1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0-1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = x0 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = x0+1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = x0+2 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+2 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b_1 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                col_index = y0 < 0 ? 0 : y0 >= {{input_width}} ? {{input_width}} - 1 : y0;
                f_1 = x0-1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0-1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = x0 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = x0+1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = x0+2 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+2 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b0 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                col_index = y0+1 < 0 ? 0 : y0+1 >= {{input_width}} ? {{input_width}} - 1 : y0+1;
                f_1 = x0-1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0-1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = x0 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = x0+1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = x0+2 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+2 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b1 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                col_index = y0+2 < 0 ? 0 : y0+2 >= {{input_width}} ? {{input_width}} - 1 : y0+2;
                f_1 = x0-1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0-1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0-1 + {{input_height}}*f)];
                f0 = x0 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0 + {{input_height}}*f)];
                f1 = x0+1 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+1 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+1 + {{input_height}}*f)];
                f2 = x0+2 < 0 ? ctx->{{output_str}}[col_index + {{input_width}}*(0 + {{input_height}}*f)] : x0+2 >= {{input_height}} ? ctx->{{output_str}}[col_index + {{input_width}}*({{input_height}} - 1 + {{input_height}}*f)] : ctx->{{output_str}}[col_index + {{input_width}}*(x0+2 + {{input_height}}*f)];
                s = x - x0;
                float register b2 = f_1*a*s*(1 + s*(s - 2)) + f0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + f1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + f2*a*s*s*(1 - s);

                s = y - y0;

                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = b_1*a*s*(1 + s*(s - 2)) + b0*(s*s*(a*(s - 1) + 2*s - 3) + 1) + b1*s*(s*(-s*(2 + a) + 2*a + 3) - a) + b2*a*s*s*(1 - s);
                ctx->tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
            }
        }
    }

    for(k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = ctx->tensor_temp[k];
    }