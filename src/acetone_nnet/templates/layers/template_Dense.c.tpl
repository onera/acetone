    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (i = 0; i < {{size}}; ++i)
    {
        dotproduct = 0;
        for (j = 0; j < {{prev_size}}; ++j)
        {
            dotproduct += {{output_str}}[j] * weights_{{name}}_{{idx}}[(j + {{prev_size}}*i)];
        }
        dotproduct += biases_{{name}}_{{idx}}[i];
        tensor_temp[i] = {{{activation_function}}};

    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }