    // {{name}}_{{idx}}
    sum = 0;
    for (i = 0; i < {{size}}; ++i)
        sum += exp({{output_str}}[i]);
    for (j = 0; j < {{size}}; ++j)
    {
        output_{{road}}[j] = exp({{output_str}}[j])/sum;
        {{#fused_layer}}
        output_{{road}}[j] = {{{fused_layer}}};
        {{/fused_layer}}
    }