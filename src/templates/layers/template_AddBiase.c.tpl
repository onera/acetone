    // {{name}}_{{idx}}{{comment}}
    for (i = 0; i < {{size}}; ++i)
    {
        output_{{road}}[i] = {{output_str}}[i] + biases_{{name}}_{{idx}}[i];
        {{^linear}}
        output_{{road}}[i] = {{{activation_function}}};
        {{/linear}}
        {{#fused_layer}}
        output_{{road}}[i] = {{{fused_layer}}};
        {{/fused_layer}}
    }