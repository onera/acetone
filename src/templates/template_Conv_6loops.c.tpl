    // {{name}}_{{idx}}
    for (int f = 0; f < {{nb_filters}}; ++f)
    {
        for (int i = 0; i < {{output_height}}; ++i)
        {
            for (int j = 0; j < {{output_width}}; ++j)
            {
                sum = 0;
                for (int c = 0; c < {{input_channels}}; ++c)
                {
                    for (int m = 0; m < {{kernel_h}}; ++m)
                    {
                        for (int n = 0; n < {{kernel_w}}; ++n)
                        {
                            int ii = i*{{strides}} + m*{{dilation_rate}} - {{pad_left}};
                            int jj = j*{{strides}} + n*{{dilation_rate}} - {{pad_top}};

                            if (ii >= 0 && ii < {{input_height}} && jj >= 0 && jj < {{input_width}})
                            {{^channels_last}}
                                sum += {{output_str}}[jj + {{input_width}}*(ii + {{input_height}}*c)] * weights_{{name}}_{{idx}}[n + {{kernel_w}}*(m + {{kernel_h}}*(c + {{input_channels}}*f))];
                            {{/channels_last}}
                            {{#channels_last}}
                                sum += {{output_str}}[(ii*{{input_width}} + jj)*{{input_channels}} + c] * weights_{{name}}_{{idx}}[n + {{kernel_w}}*(m + {{kernel_h}}*(c + {{input_channels}}*f))];
                            {{/channels_last}}
                        }
                    }
                }
                sum += biases_{{name}}_{{idx}}[f];
                {{^fused_layer}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{activation_function}};
                {{/fused_layer}}
                {{#fused_layer}}
                    {{^linear}}
                sum = {{activation_function}}
                    {{/linear}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{fused_layer}};
                {{/fused_layer}}
            }
        }
    }
{{^channels_last}}
    for (int k = 0; k < {{size}}; ++k)
    {
        
        output_{{road}}[k] = tensor_temp[k];
    }
{{/channels_last}}
{{#channels_last}}
    for(int f = 0; f < {{input_channels}}; ++f)
    {
        for (int i = 0; i < {{input_height}}; ++i)
        {
            for (int j = 0; j < {{input_width}};  ++j)
            {
                output_{{road}}[(i*{{input_width}} + j)*{{input_channels}} + f] = tensor_temp[(f*{{input_height}} + i)*{{input_width}} + j];
            }
        }
    }
{{/channels_last}}