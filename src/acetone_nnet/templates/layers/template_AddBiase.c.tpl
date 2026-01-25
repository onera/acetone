    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (i = 0; i < {{size}}; ++i)
    {
        ctx->output_{{road}}[i] = ctx->{{output_str}}[i] + biases_{{name}}_{{idx}}[i];
        {{^linear}}
        ctx->output_{{road}}[i] = {{{activation_function}}};
        {{/linear}}
        {{#activation_function}}
        ctx->output_{{road}}[i] = {{{activation_function}}};
        {{/activation_function}}
    }