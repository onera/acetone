    // gemm_tn
    for (i = 0; i < {{m}}; ++i)
    {
        for (p = 0; p < {{k}}; ++p)
        {
            float register weight = {{A}}[p*{{ldA}}+i];
            for (j = 0; j < {{n}}; ++j)
            {
                tensor_temp[i*{{ldC}}+j] += weight * {{#direct}}*{{/direct}}({{B}}[p*{{ldB}}+j]);
            }
            for(j = 0; j < {{n}}; ++j){
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
    }