    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                int new_f = f - {{pads_front}};
                int new_i = i - {{pads_top}};
                int new_j = j - {{pads_left}};

                {{{change_indice}}}

                tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{output_str}}[new_j + {{input_width}} * (new_i + {{input_height}} * new_f)];
                {{^linear}}
                tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{{activation_function}}};
                {{/linear}}
                {{#activation_function}}
                tensor_temp[j + {{output_width}} * (i + {{output_height}} * f)] = {{{activation_function}}};
                {{/activation_function}}
            }
        }
    }
    for(k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }