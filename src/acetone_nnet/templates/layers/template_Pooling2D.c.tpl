    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (f = 0; f < {{input_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                {{{update_local_vars}}}
                for (h = 0; h < {{pool_size}}; ++h)
                {
                    for (w = 0; w < {{pool_size}}; ++w)
                    {
                        int ii = i*{{strides}} + h - {{pad_left}};
                        int jj = j*{{strides}} + w - {{pad_top}};
                        if (ii >= 0 && ii < {{input_height}} && jj >= 0 && jj < {{input_width}})
                        {
                            {{{specific_function}}}
                        }
                    }
                }
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
            }
        }
    }

    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }