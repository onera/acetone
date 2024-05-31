    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; f++)
    {
        for (i = 0; i < {{output_height}}; i++)
        {
            for (j = 0; j < {{output_width}}; j++)
            {
                {{#channels}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[j + {{input_width}}*(i + {{input_height}}*indice_Gather_{{idx}}[j + {{output_width}}*(i + {{output_height}}*f)])];
                {{/channels}}
                {{#heights}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[j + {{input_width}}*(indice_Gather_{{idx}}[j + {{output_width}}*(i + {{output_height}}*f)] + {{input_height}}*f)];
                {{/heights}}
                {{#widths}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[indice_Gather_{{idx}}[j + {{output_width}}*(i + {{output_height}}*f)] + {{input_width}}*(i + {{input_height}}*f)];
                {{/widths}}
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_function}}};
    }