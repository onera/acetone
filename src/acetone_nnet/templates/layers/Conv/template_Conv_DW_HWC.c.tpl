    /* {{name}}_{{idx}} {{KH}}x{{KW}} depth-wise {{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
        * Input [H][W][C]:[{{H_in}}][{{W_in}}][{{C}}]
        * Output [H][W][K]:[{{H_out}}][{{W_out}}][{{K}}]
        * Kernel [KH][KW]:[{{KH}}][{{KW}}]
        * stride:{{STRIDE}} pad:{{PAD}}
        */

    for (int oh = 0; oh < {{H_out}}; ++oh) {
        const int h_start_idx = oh * {{STRIDE}} - {{PAD}};
        const int kh_start = (0 > -h_start_idx) ? 0 : -h_start_idx;
        const int kh_end = ({{KH}} < {{H_in}} - h_start_idx) ? {{KH}} : ({{H_in}} - h_start_idx);

        for (int ow = 0; ow < {{W_out}}; ++ow) {
            // Accès direct avec restrict pour l'écriture
            float * restrict out_ptr = (float * restrict)&ctx->tensor_temp[(oh * {{W_out}} * {{K}}) + (ow * {{K}})];
            #pragma omp simd
            for (int k = 0; k < {{K}}; ++k) {
                out_ptr[k] = ((const float * restrict)biases_{{name}}_{{idx}})[k];
            }

            const int w_start_idx = ow * {{STRIDE}} - {{PAD}};
            const int kw_start = (0 > -w_start_idx) ? 0 : -w_start_idx;
            const int kw_end = ({{KW}} < {{W_in}} - w_start_idx) ? {{KW}} : ({{W_in}} - w_start_idx);

            for (int kh = kh_start; kh < kh_end; ++kh) {
                const int ih = h_start_idx + kh;
                for (int kw = kw_start; kw < kw_end; ++kw) {
                    const int iw = w_start_idx + kw;
                    const float * restrict in_ptr = (const float * restrict)&ctx->{{output_str}}[(ih * {{W_in}} * {{K}}) + (iw * {{K}})];
                    const float * restrict w_ptr  = (const float * restrict)&weights_{{name}}_{{idx}}[(kh * {{KW}} * {{K}}) + (kw * {{K}})];

                    #pragma omp simd
                    for (int k = 0; k < {{K}}; ++k) {
                        out_ptr[k] += in_ptr[k] * w_ptr[k];
                    }
                }
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        {{#activation_function}}
        ctx->output_{{road}}[k] = {{{activation_function}}};
        {{/activation_function}}
        {{^activation_function}}
        ctx->output_{{road}}[k] = ctx->tensor_temp[k];
        {{/activation_function}}
    }

