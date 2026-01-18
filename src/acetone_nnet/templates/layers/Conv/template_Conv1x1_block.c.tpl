    // Conv2D 1x1 :
    // Input HWC [{{IN_H}}][{{IN_W}}][{{C_IN}}]
    // Output channel [{{C_OUT}}] bloc [{{C_OUT_BLOCK}}]
    // Stride: {{STRIDE}}    
    for (int oh = 0; oh < {{IN_H}} / {{STRIDE}}; ++oh) {
        for (int ow = 0; ow < {{IN_W}} / {{STRIDE}}; ++ow) {
            
            // Calculate input offset based on stride
            int ih = oh * {{STRIDE}};
            int iw = ow * {{STRIDE}};
            const float* restrict pixel_in = &{{output_str}}[(ih * {{IN_W}} + iw) * {{C_IN}}];
            float* restrict pixel_out = &tensor_temp[(oh * ({{IN_W}} / {{STRIDE}}) + ow) * {{C_OUT}}];

            for (int oc_b = 0; oc_b < {{C_OUT}}; oc_b += {{C_OUT_BLOCK}}) {
                
                float sum_array[{{C_OUT_BLOCK}}] __attribute__((aligned(64)));
                for (int k = 0; k < {{C_OUT_BLOCK}}; ++k)
                    sum_array[k] = biases_{{name}}_{{idx}}[oc_b + k];

                for (int ic = 0; ic < {{C_IN}}; ++ic) {
                    float val_in = pixel_in[ic];

                    // Weights layout: [Groups of BK][C_IN][BK]
                    const float* restrict w_ptr = &weights_{{name}}_{{idx}}[(oc_b / {{C_OUT_BLOCK}}) * {{C_IN}} * {{C_OUT_BLOCK}} + (ic * {{C_OUT_BLOCK}})];

                    #pragma omp simd
                    for (int b = 0; b < {{C_OUT_BLOCK}}; ++b) {
                        sum_array[b] += val_in * w_ptr[b];
                    }
                }

                // Store blocked result to NHWC output
                {{#activation_function}}
                for (int b = 0; b < {{C_OUT_BLOCK}}; ++b)
                           sum_array[b] = {{{activation_function}}};
                {{/activation_function}}
                for (int b = 0; b < {{C_OUT_BLOCK}}; ++b)
                    pixel_out[oc_b + b] = sum_array[b];
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }
