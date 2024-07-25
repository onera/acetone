// Convolution kernel for layer {{layer_id}}
{

    /**
     * @brief Naive convolution implementation
     * @tparam T input and output type
     */
    template <typename T>
    struct _convolution_operation
    {
        /**
         * @brief Naive convolution implementation
         * @param channel_count number of channels in input and kernels
         * @param kernel_count number of kernels
         * @param input_shape shape of input {height, width}
         * @param kernel_shape shape of kernels {height, width}
         * @param output_shape shape of output {height, width}
         * @param stride step between input elements for consecutive kernel applications, homogeneous across input dimensions
         * @param dilation step between input elements for consecutive kernel elements, homogeneous across input dimensions
         * @param padding added elements around each input dimension {height, width}
         * @param kernels
         * @param input
         * @param output
         */
         static inline void call(
            const size_t channel_count,
            const size_t kernel_count,
            const size_t kernel_shape[2],
            const size_t input_shape[2],
            const size_t output_shape[2],
            const size_t stride,
            const size_t dilation,
            const size_t padding[2],

            const T kernels[],
            const T input[],
            const T biases[],
            T output[])
        {
            for (size_t f = 0; f < kernel_count; ++f)
            {
                for (size_t i = 0; i < output_shape[0]; ++i)
                {
                    for (size_t j = 0; j < output_shape[1]; ++j)
                    {
                        T sum = 0;
                        for (size_t p = 0; p < channel_count; ++p)
                        {
                            for (size_t h = 0; h < kernel_shape[0]; ++h)
                            {
                                for (size_t w = 0; w < kernel_shape[1]; ++w)
                                {
                                    size_t ii = (i * stride) + (h * dilation) - padding[0];
                                    size_t jj = (j * stride) + (w * dilation) - padding[1];

                                    if (ii >= 0 && ii < input_shape[0] && jj >= 0 && jj < input_shape[1])
                                        sum += input[jj + input_shape[1]*(ii + input_shape[0]*p)] * kernels[w + kernel_shape[1]*(h + kernel_shape[0]*(p + channel_count*f))];
                                }
                            }
                        }
                        output[j + output_shape[1] * (i + output_shape[0] * f)] = sum + biases[f];
                    }
                }
            }
        }
    }


    size_t kernel_shape[2] = { {{kernel_height}}, {{kernel_width}} };
    size_t input_shape[2]  = { {{input_height}},  {{input_width}}  };
    size_t output_shape[2] = { {{output_height}}, {{output_width}} };

    {{tensor_type}} padding[2] = { {{padding_top}}, {{padding_left}} };
    {{tensor_type}} kernels[] = {{kernel_var_name}};
    {{tensor_type}} input[] = {{input_var_name}};
    {{tensor_type}} biases[] = {{biases_var_name}};
    {{tensor_type}} output[{{output_size}}];

    _convolution_operation<{{tensor_type}}.call(
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
        output_{{road}}[k] = {{activation_function_src}}; // activation(output[k]);
    }

}