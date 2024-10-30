    // gemm target
    {
        int register mc, nc, kc;
        int register i, j, l;
        for (j = 0; j < {{n}}; j += NC) 
        {
            nc = min( {{n}} - j, NC );
            for (l = 0; l < {{k}}; l += KC) 
            {
                kc = min( {{k}} - l, KC);
                pack_B(kc, nc, &{{B}}[l * {{ldB}} + j], {{ldB}}, 1, packed_B);
                for (i = 0; i < {{m}}; i += MC) 
                {
                    mc = min ({{m}} - i, MC);
                    pack_A(mc,kc, &{{A}}[i * {{ldA}} + l], {{ldA}}, 1, packed_A);
                    sgemm_macro_kernel(mc, nc, kc, &tensor_temp[i*{{ldC}}+j], {{ldC}}, 1);
                }
            }
        }
    }
    // Biases computation
    for (i = 0; i < {{m}}; ++i)
    {
        for(j = 0; j < {{n}}; ++j)
        {
            register float output = tensor_temp[i*{{ldC}} + j];
            output += biases_{{name}}_{{idx}}[i];
        {{^fused_layer}}
            tensor_temp[i*{{ldC}} + j] = {{{activation_function}}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
            output = {{{activation_function}}};
            {{/linear}}
            tensor_temp[i*{{ldC}} + j] = {{{fused_layer}}};
        {{/fused_layer}}
        }
    }
