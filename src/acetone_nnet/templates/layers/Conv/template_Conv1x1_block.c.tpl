    /* {{name}}_{{idx}} 1x1 - Format HWC {{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
     * Input HWC [{{H_in}}][{{W_in}}][{{C}}]
     * Output channel [{{K}}] = [{{NB_BLOCK_K}}x{{BLOCK_K}}]
     * Stride: {{STRIDE}}
     */
    for (int oh = 0; oh < {{H_in}} / {{STRIDE}}; ++oh) {
        for (int ow = 0; ow < {{W_in}} / {{STRIDE}}; ++ow) {            
            const float* restrict pixel_in = &{{output_str}}[(oh * {{STRIDE}} * {{W_in}} + ow * {{STRIDE}}) * {{C}}];
            float* restrict o_ptr = &tensor_temp[(oh * ({{W_in}} / {{STRIDE}}) + ow) * {{K}}];

            for (int ok_b = 0; ok_b < {{K}}; ok_b += {{BLOCK_K}}) {
                
                for (int k = 0; k < {{BLOCK_K}}; ++k)
                    o_ptr[ok_b+k] = biases_{{name}}_{{idx}}[ok_b + k];

                for (int ic = 0; ic < {{C}}; ++ic) {
                    float val_in = pixel_in[ic];
                    // Weights layout: [NB_BK][C][BK] = [{{NB_BLOCK_K}}][{{C}}][{{BLOCK_K}}]
                    const float* restrict w_ptr = &weights_{{name}}_{{idx}}[(ok_b / {{BLOCK_K}}) * {{C}} * {{BLOCK_K}} + (ic * {{BLOCK_K}})];
                    #pragma omp simd
                    for (int k = 0; k < {{BLOCK_K}}; ++k)
                        o_ptr[ok_b+k] += val_in * w_ptr[k];
                }
            }
       }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        {{#activation_function}}
        output_{{road}}[k] = {{{activation_function}}};
        {{/activation_function}}
        {{^activation_function}}
        output_{{road}}[k] = tensor_temp[k];
        {{/activation_function}}
    }
