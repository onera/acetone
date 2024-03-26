    // {{name}}_{{idx}}
    sum = 0;
    for (int i = 0; i < {{size}}; ++i)
        sum += exp({{output_str}}[i]);
    for (int j = 0; j < {{size}}; ++j)
    {
        output_{{road}}[j] = exp({{output_str}}[j])/sum;
        {{#fused_layer}}
        output_{{road}}[j] = {{fused_layer}};
        {{/fused_layer}}
    }