    // {{name}}_{{idx}}{{comment}} {{#channels_last}}HWC layout{{/channels_last}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    {
        {{#is_avgpool}}
        // Calcul de l'aire inverse déplacé au début (invariant)
        float inv_area = 1.0f / (float)({{pool_size}} * {{pool_size}});
        {{/is_avgpool}}
        for (int oh = 0; oh < {{output_height}}; ++oh) {
            int temp_h_start = {{pad_left}} - oh * {{strides}};
            int kh_start = (0 > temp_h_start) ? 0 : temp_h_start;
            
            int temp_h_end = {{input_height}} + {{pad_left}} - oh * {{strides}};
            int kh_end = ({{pool_size}} < temp_h_end) ? {{pool_size}} : temp_h_end;

            for (int ow = 0; ow < {{output_width}}; ++ow) {
                int temp_w_start = {{pad_left}} - ow * {{strides}};
                int kw_start = (0 > temp_w_start) ? 0 : temp_w_start;
                
                int temp_w_end = {{input_width}} + {{pad_left}} - ow * {{strides}};
                int kw_end = ({{pool_size}} < temp_w_end) ? {{pool_size}} : temp_w_end;
                float* out_ptr = &tensor_temp[(oh * {{output_width}} + ow) * {{input_channels}}];

                // Initialize channel accumulators
                for (int c = 0; c < {{input_channels}}; ++c) {
                    {{#is_maxpool}}out_ptr[c] = -INFINITY;{{/is_maxpool}}
                    {{#is_avgpool}}out_ptr[c] = 0.0f;{{/is_avgpool}}
                }

                // Spatial window loops
                for (int kh = kh_start; kh < kh_end; ++kh) {
                    int ih = oh * {{strides}} + kh - {{pad_left}};
                    for (int kw = kw_start; kw < kw_end; ++kw) {
                        int iw = ow * {{strides}} + kw - {{pad_left}};
                        
                        const float* in_ptr = &{{output_str}}[(ih * {{input_width}} + iw) * {{input_channels}}];

                        // Innermost Channel Loop 
                        for (int c = 0; c < {{input_channels}}; ++c) {
                            {{#is_maxpool}}
                            out_ptr[c] = (in_ptr[c] > out_ptr[c]) ? in_ptr[c] : out_ptr[c];
                            {{/is_maxpool}}
                            {{#is_avgpool}}
                            out_ptr[c] += in_ptr[c];
                            {{/is_avgpool}}
                        }
                    }
                }

                {{#is_avgpool}}
                // Averaging
                for (int c = 0; c < {{input_channels}}; ++c) {
                    out_ptr[c] *= inv_area;
                }
                {{/is_avgpool}}
                {{#activation_function}}
                // Activation
                for (int c = 0; c < {{input_channels}}; ++c) {
                    out_ptr[c] = {{{activation_function}}};
                }
                {{/activation_function}}
            }
        }
        for (k = 0; k < {{size}}; ++k)
        {
            output_{{road}}[k] = tensor_temp[k];
        }
    }