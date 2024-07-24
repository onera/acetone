// Convolution kernel for layer {{layer_id}}
{
    size_t kernel_shape[2] = { {{kernel_height}}, {{kernel_width}} };
    size_t input_shape[2]  = { {{input_height}},  {{input_width}}  };
    size_t output_shape[2] = { {{output_height}}, {{output_width}} };

    {{tensor_type}} padding[2] = { {{padding_top}}, {{padding_left}} };
    {{tensor_type}} kernels[] = {{kernel_var_name}};
    {{tensor_type}} input[] = {{input_var_name}};
    {{tensor_type}} biases[] = {{biases_var_name}};
    {{tensor_type}} output[{{output_size}}];

    perform_convolution<{{tensor_type}}>(
        {{channel_count}},
        {{kernel_count}},
        kernel_shape,
        input_shape,
        output_shape,
        {{stride}},
        {{dilation}},
        padding,
        kernels,
        input,
        biases,
        output
    );

    // TODO Deal with fused layers and activation functions
    //    {{#fused_layer}}
    //    sum = {{{activation_function}}};
    //    tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{{fused_layer}}};
    //    {{/fused_layer}}

    for (k = 0; k < {{output_size}}; ++k)
    {
        output_{{road}}[k] = {{activation_function}}; // activation(output[k]);
    }

}