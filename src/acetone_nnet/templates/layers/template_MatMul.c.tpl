    // {{name}}_{{idx}}{{comment}}
    for (i = 0; i < {{size}}; ++i)
    {
        dotproduct = 0;
        for (j = 0; j < {{prev_size}}; ++j)
        {
            {{#side}}
            dotproduct += weights_{{name}}_{{idx}}[(j + {{prev_size}}*i)] * {{output_str}}[j];
            {{/side}}
            {{^side}}
            dotproduct += {{output_str}}[j] * weights_{{name}}_{{idx}}[(j + {{prev_size}}*i)];
            {{/side}}
        }
        {{^fused_layer}}
        tensor_temp[i] = {{{activation_function}}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
        dotproduct = {{{activation_function}};}
            {{/linear}}
        tensor_temp[i] = {{{fused_layer}}};
        {{/fused_layer}}
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }