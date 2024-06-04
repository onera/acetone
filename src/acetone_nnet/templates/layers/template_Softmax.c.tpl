    // {{name}}_{{idx}}
    {{#1D}}
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
    {{/1D}}
    {{^1D}}
    for (f = 0; f < {{sum_dimension_1}}; ++f)
    {
        for (i = 0; i < {{sum_dimension_2}}; ++i)
        {
            tensor_temp[{{reduced_position_1}}] = 0;
            for (j = 0; j < {{reduced_dimension}}; ++j)
            {
                tensor_temp[{{reduced_position_1}}] += exp({{output_str}}[{{reduced_position_2}}]);
            }
        }
    }

    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                output_{{road}}[j + {{output_width}}*(i +  {{output_height}}*f)] = exp({{output_str}}[j + {{output_width}}*(i +  {{output_height}}*f)])/tensor_temp[{{softmax_indice}}];
            }
        }
    }
    {{/1D}}