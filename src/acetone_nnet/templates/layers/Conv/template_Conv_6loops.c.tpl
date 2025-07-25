    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (f = 0; f < {{nb_filters}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                sum = 0;
                for (p = 0; p < {{input_channels}}; ++p)
                {
                    for (h = 0; h < {{kernel_h}}; ++h)
                    {
                        for (w = 0; w < {{kernel_w}}; ++w)
                        {
                            int ii = i*{{strides}} + h*{{dilation_rate}} - {{pad_top}};
                            int jj = j*{{strides}} + w*{{dilation_rate}} - {{pad_left}};

                            if (ii >= 0 && ii < {{input_height}} && jj >= 0 && jj < {{input_width}})
                                sum += {{output_str}}[jj + {{input_width}}*(ii + {{input_height}}*p)] * weights_{{name}}_{{idx}}[w + {{kernel_w}}*(h + {{kernel_h}}*(p + {{input_channels}}*f))];
                        }
                    }
                }
                sum += biases_{{name}}_{{idx}}[f];
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{activation_function}}};
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }