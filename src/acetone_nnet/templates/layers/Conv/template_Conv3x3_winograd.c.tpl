    /**
    * Conv2D 3x3 Noyau Winograd F(2,2, 3,3) - Format HWC
    * In HWC: [{{in_h}}][{{in_w}}][{{in_c}}]
    * Out HWK: [{{out_h}}][{{out_w}}][{{out_c}}]
    * Bloc K : {{bk_size}}
    * Pad HW: [{{pad_h}}][{{pad_w}}] 
    */
    for (int k = 0; k < {{out_c}}; k += {{bk_size}}) {
        /* Parcours direct de l'image par tuiles de 2x2 */
        for (int oh = 0; oh < ({{out_h}} / 2) * 2; oh += 2){
        for (int ow = 0; ow < ({{out_w}} / 2) * 2; ow += 2) {
            /* Accumulateurs pour les 16 points du domaine Winograd sur BK canaux */
            float acc[4][4][{{bk_size}}] = {0};

            for (int ic = 0; ic < {{in_c}}; ic++) {
                float d[4][4];
                
                /* Chargement du bloc 4x4 d'entrée avec gestion du padding sans IF */
                for(int r = 0; r < 4; ++r) {
                    int ih = oh + r - {{pad_h}};
                    const float* row_ptr = &{{output_str}}[ih * {{in_w}} * {{in_c}} + ic];
                    int valid_h = (ih >= 0 && ih < {{in_h}});
                    
                    for(int c = 0; c < 4; ++c) {
                        int iw = ow + c - {{pad_w}};
                        int valid_w = (iw >= 0 && iw < {{in_w}});
                        d[r][c] = (valid_h & valid_w) ? row_ptr[iw * {{in_c}}] : 0.0f;
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
                        const float* __restrict w0 = &weights_{{name}}_{{idx}}[((((k/{{bk_size}})*4 + i)*4 + 0)*{{in_c}} + ic)*{{bk_size}}];
                        for(int bk = 0; bk < {{bk_size}}; bk++) acc[i][0][bk] += (temp[i][0] - temp[i][2]) * w0[bk];
                    }
                    {
                        const float* __restrict w1 = &weights_{{name}}_{{idx}}[((((k/{{bk_size}})*4 + i)*4 + 1)*{{in_c}} + ic)*{{bk_size}}];
                        for(int bk = 0; bk < {{bk_size}}; bk++) acc[i][1][bk] += (temp[i][1] + temp[i][2]) * w1[bk];
                    }
                    {
                        const float* __restrict w2 = &weights_{{name}}_{{idx}}[((((k/{{bk_size}})*4 + i)*4 + 2)*{{in_c}} + ic)*{{bk_size}}];
                        for(int bk = 0; bk < {{bk_size}}; bk++) acc[i][2][bk] += (temp[i][2] - temp[i][1]) * w2[bk];
                    }
                    {
                        const float* __restrict w3 = &weights_{{name}}_{{idx}}[((((k/{{bk_size}})*4 + i)*4 + 3)*{{in_c}} + ic)*{{bk_size}}];
                        for(int bk = 0; bk < {{bk_size}}; bk++) acc[i][3][bk] += (temp[i][1] - temp[i][3]) * w3[bk];
                    }
                }
            }

            float m_temp[2][4][{{bk_size}}] __attribute__((aligned(64)));
            /* 3. Transformation de sortie (AT * M * A) + Biais + Activation */
            for (int bk = 0; bk < {{bk_size}}; bk++) {
                m_temp[0][0][bk] = acc[0][0][bk] + acc[1][0][bk] + acc[2][0][bk];
                m_temp[1][0][bk] = acc[1][0][bk] - acc[2][0][bk] - acc[3][0][bk];
                m_temp[0][1][bk] = acc[0][1][bk] + acc[1][1][bk] + acc[2][1][bk];
                m_temp[1][1][bk] = acc[1][1][bk] - acc[2][1][bk] - acc[3][1][bk];
                m_temp[0][2][bk] = acc[0][2][bk] + acc[1][2][bk] + acc[2][2][bk];
                m_temp[1][2][bk] = acc[1][2][bk] - acc[2][2][bk] - acc[3][2][bk];
                m_temp[0][3][bk] = acc[0][3][bk] + acc[1][3][bk] + acc[2][3][bk];
                m_temp[1][3][bk] = acc[1][3][bk] - acc[2][3][bk] - acc[3][3][bk];

            }
            {
                float *restrict out_ptr = &{{output_name}}[((oh + 0) * {{out_w}} + (ow + 0)) * {{out_c}} + k];
                for (int bk = 0; bk < {{bk_size}}; bk++)
                out_ptr[bk] = m_temp[0][0][bk] + m_temp[0][1][bk] + m_temp[0][2][bk] + biases_{{name}}_{{idx}}[k + bk];
            }
            {    
                float *restrict out_ptr = &{{output_name}}[((oh + 0) * {{out_w}} + (ow + 1)) * {{out_c}} + k];
                for (int bk = 0; bk < {{bk_size}}; bk++)
                    out_ptr[bk] = m_temp[0][1][bk] - m_temp[0][2][bk] - m_temp[0][3][bk] + biases_{{name}}_{{idx}}[k + bk];
            }
            {
                float *restrict out_ptr = &{{output_name}}[((oh + 1) * {{out_w}} + (ow + 0)) * {{out_c}} + k];
                for (int bk = 0; bk < {{bk_size}}; bk++)
                    out_ptr[bk] = m_temp[1][0][bk] + m_temp[1][1][bk] + m_temp[1][2][bk] + biases_{{name}}_{{idx}}[k + bk];
            }
            {    
                float *restrict out_ptr = &{{output_name}}[((oh + 1) * {{out_w}} + (ow + 1)) * {{out_c}} + k];
                for (int bk = 0; bk < {{bk_size}}; bk++)
                    out_ptr[bk] = m_temp[1][1][bk] - m_temp[1][2][bk] - m_temp[1][3][bk] + biases_{{name}}_{{idx}}[k + bk];
            }        
        }
    }
    }
