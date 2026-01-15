    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    // HI:{{H_in}} WI:{{W_in}} C:{{C}}
    // HO:{{H_out}} WO:{{W_out}} K:{{K}}
    // KH:{{KH}} KW:{{KW}} 
    // stride:{{STRIDE}} pad:{{PAD}}
    // NB_BLOCK_C:{{NB_BLOCK_C}} NB_BLOCK_K:{{NB_BLOCK_K}}
    // BLOCK_C:{{BLOCK_C}} BLOCK_K:{{BLOCK_K}}

    // Optimized Kernel
    for (int bk = 0; bk < {{NB_BLOCK_K}}; ++bk) {
        const int k_base = bk * {{BLOCK_K}};
        for (int oh = 0; oh < {{H_out}}; ++oh) {
            const int kh_s = ({{PAD}} - oh * {{STRIDE}} > 0) ? ({{PAD}} - oh * {{STRIDE}}) : 0;
            const int kh_e = ({{KH}} < ({{H_in}} + {{PAD}} - oh * {{STRIDE}})) ? {{KH}} : ({{H_in}} + {{PAD}} - oh * {{STRIDE}});
            for (int ow = 0; ow < {{W_out}}; ++ow) {
                const int kw_s = ({{PAD}} - ow * {{STRIDE}} > 0) ? ({{PAD}} - ow * {{STRIDE}}) : 0;
                const int kw_e = ({{KW}} < ({{W_in}} + {{PAD}} - ow * {{STRIDE}})) ? {{KW}} : ({{W_in}} + {{PAD}} - ow * {{STRIDE}});
                
                float* restrict o_ptr = &tensor_temp[(oh * {{W_out}} * {{K}}) + (ow * {{K}}) + k_base];
                for (int k = 0; k < {{BLOCK_K}}; ++k) o_ptr[k] = biases_{{name}}_{{idx}}[k_base + k];

                for (int bc = 0; bc < {{NB_BLOCK_C}}; ++bc) {
                    const int c_base = bc * {{BLOCK_C}};
                    for (int kh = kh_s; kh < kh_e; ++kh) {
                        const int ih = oh * {{STRIDE}} - {{PAD}} + kh;
                        for (int kw = kw_s; kw < kw_e; ++kw) {
                            const int iw = ow * {{STRIDE}} - {{PAD}} + kw;
                            const float* restrict i_ptr = &{{output_str}}[(ih * {{W_in}} + iw) * {{IC}} + c_base];
                            const float* restrict w_ptr_base = &weights_{{name}}_{{idx}}[((((bk * {{NB_BLOCK_C}} + bc) * {{KH}} + kh) * {{KW}} + kw) * {{BLOCK_C}} * {{BLOCK_K}})];
                            for (int c = 0; c < {{BLOCK_C}}; ++c) {
                                float iv = i_ptr[c];
                                #pragma omp simd
                                for (int k = 0; k < {{BLOCK_K}}; ++k) o_ptr[k] += iv * w_ptr_base[c * {{BLOCK_K}} + k];
                            }
                        }
                    }
                }
                {{#activation_function}}
                for (int k = 0; k < {{BLOCK_K}}; ++k) o_ptr[k] = {{{activation_function}}};
                {{/activation_function}}
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }
