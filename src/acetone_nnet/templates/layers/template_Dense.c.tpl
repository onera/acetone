    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (i = 0; i < {{size}}; ++i)
    {
        dotproduct = 0;
        for (j = 0; j < {{prev_size}}; ++j)
        {
            dotproduct += ctx->{{output_str}}[j] * weights_{{name}}_{{idx}}[((j * {{size}}) + i)];
        }
        dotproduct += biases_{{name}}_{{idx}}[i];
        ctx->tensor_temp[i] = {{{activation_function}}};

    }
    for (k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = ctx->tensor_temp[k];
    }