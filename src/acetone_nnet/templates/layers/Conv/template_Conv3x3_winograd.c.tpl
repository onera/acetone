    /**
    * {{name}}_{{idx}} 3x3 Winograd F(2,2, 3,3) - Format HWC {{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    * In HWC: [{{H_in}}][{{W_in}}][{{C}}]
    * Out HWK: [{{H_out}}][{{W_out}}][{{K}}]
    * Bloc K : {{BLOCK_K}}
    * Pad : [{{PAD}}] Stride : 1
    */
    for (int k_base = 0; k_base < {{K}}; k_base += {{BLOCK_K}}){
        /* Parcours direct de l'image par tuiles de 2x2 */
        for (int oh = 0; oh < ({{H_out}} / 2) * 2; oh += 2){
        for (int ow = 0; ow < ({{W_out}} / 2) * 2; ow += 2){
            /* Accumulateurs pour les 16 points du domaine Winograd sur BK canaux */
            float acc[4][4][{{BLOCK_K}}] = {0};

            for (int ic = 0; ic < {{C}}; ic++) {
                float d[4][4];
                
                /* Chargement du bloc 4x4 d'entrée avec gestion du padding sans IF */
                for(int r = 0; r < 4; ++r) {
                    int ih = oh + r - {{PAD}};
                    const float* row_ptr = &{{output_str}}[ih * {{W_in}} * {{C}} + ic];
                    int valid_h = (ih >= 0 && ih < {{H_in}});
                    
                    for(int c = 0; c < 4; ++c) {
                        int iw = ow + c - {{PAD}};
                        int valid_w = (iw >= 0 && iw < {{W_in}});
                        d[r][c] = (valid_h & valid_w) ? row_ptr[iw * {{C}}] : 0.0f;
                    }
                }

                /* 1. Transformation de l'entrée (Data Transform BT * d * B) */
                float temp[4][4];
                for(int i = 0; i < 4; i++) {
                    temp[0][i] = d[0][i] - d[2][i];
                    temp[1][i] = d[1][i] + d[2][i];
                    temp[2][i] = d[2][i] - d[1][i];
                    temp[3][i] = d[1][i] - d[3][i];
                }

                /* 2. Transformation horizontale + Multiplication + Accumulation */
                for(int i = 0; i < 4; i++) {
                    {
                        const float* __restrict w0 = &weights_{{name}}_{{idx}}[((((k_base/{{BLOCK_K}})*4 + i)*4 + 0)*{{C}} + ic)*{{BLOCK_K}}];
                        for(int k = 0; k < {{BLOCK_K}}; k++) acc[i][0][k] += (temp[i][0] - temp[i][2]) * w0[k];
                    }
                    {
                        const float* __restrict w1 = &weights_{{name}}_{{idx}}[((((k_base/{{BLOCK_K}})*4 + i)*4 + 1)*{{C}} + ic)*{{BLOCK_K}}];
                        for(int k = 0; k < {{BLOCK_K}}; k++) acc[i][1][k] += (temp[i][1] + temp[i][2]) * w1[k];
                    }
                    {
                        const float* __restrict w2 = &weights_{{name}}_{{idx}}[((((k_base/{{BLOCK_K}})*4 + i)*4 + 2)*{{C}} + ic)*{{BLOCK_K}}];
                        for(int k = 0; k < {{BLOCK_K}}; k++) acc[i][2][k] += (temp[i][2] - temp[i][1]) * w2[k];
                    }
                    {
                        const float* __restrict w3 = &weights_{{name}}_{{idx}}[((((k_base/{{BLOCK_K}})*4 + i)*4 + 3)*{{C}} + ic)*{{BLOCK_K}}];
                        for(int k = 0; k < {{BLOCK_K}}; k++) acc[i][3][k] += (temp[i][1] - temp[i][3]) * w3[k];
                    }
                }
            }

            float m_temp[2][4][{{BLOCK_K}}] __attribute__((aligned(64)));
            /* 3. Transformation de sortie (AT * M * A) + Biais + Activation */
            for (int k = 0; k < {{BLOCK_K}}; k++) {
                m_temp[0][0][k] = acc[0][0][k] + acc[1][0][k] + acc[2][0][k];
                m_temp[1][0][k] = acc[1][0][k] - acc[2][0][k] - acc[3][0][k];
                m_temp[0][1][k] = acc[0][1][k] + acc[1][1][k] + acc[2][1][k];
                m_temp[1][1][k] = acc[1][1][k] - acc[2][1][k] - acc[3][1][k];
                m_temp[0][2][k] = acc[0][2][k] + acc[1][2][k] + acc[2][2][k];
                m_temp[1][2][k] = acc[1][2][k] - acc[2][2][k] - acc[3][2][k];
                m_temp[0][3][k] = acc[0][3][k] + acc[1][3][k] + acc[2][3][k];
                m_temp[1][3][k] = acc[1][3][k] - acc[2][3][k] - acc[3][3][k];

            }
            {
                float *restrict o_ptr = &tensor_temp[((oh + 0) * {{W_out}} + (ow + 0)) * {{K}} + k_base];
                for (int k = 0; k < {{BLOCK_K}}; k++)
                    o_ptr[k] = m_temp[0][0][k] + m_temp[0][1][k] + m_temp[0][2][k] + biases_{{name}}_{{idx}}[k_base + k];
            }
            {    
                float *restrict o_ptr = &tensor_temp[((oh + 0) * {{W_out}} + (ow + 1)) * {{K}} + k_base];
                for (int k = 0; k < {{BLOCK_K}}; k++)
                    o_ptr[k] = m_temp[0][1][k] - m_temp[0][2][k] - m_temp[0][3][k] + biases_{{name}}_{{idx}}[k_base + k];
            }
            {
                float *restrict o_ptr = &tensor_temp[((oh + 1) * {{W_out}} + (ow + 0)) * {{K}} + k_base];
                for (int k = 0; k < {{BLOCK_K}}; k++)
                    o_ptr[k] = m_temp[1][0][k] + m_temp[1][1][k] + m_temp[1][2][k] + biases_{{name}}_{{idx}}[k_base + k];
            }
            {    
                float *restrict o_ptr = &tensor_temp[((oh + 1) * {{W_out}} + (ow + 1)) * {{K}} + k_base];
                for (int k = 0; k < {{BLOCK_K}}; k++)
                    o_ptr[k] = m_temp[1][1][k] - m_temp[1][2][k] - m_temp[1][3][k] + biases_{{name}}_{{idx}}[k_base + k];
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