    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; f++)
    {
        for (i = 0; i < {{output_height}}; i++)
        {
            for (j = 0; j < {{output_width}}; j++)
            {
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[{{a}} + {{input_width}}*({{b}} + {{input_height}}*{{c}})];
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_function}}};
    }