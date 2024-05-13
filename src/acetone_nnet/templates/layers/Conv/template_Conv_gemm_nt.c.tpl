    // gemm_nt
    for (i = 0; i < {{m}}; ++i)
    {
        for (j = 0; j < {{n}}; ++j)
        {
            register float output =0;
            for (p = 0; p < {{k}}; ++p)
            {
                output += {{A}}[i*{{ldA}}+p]* {{#direct}}*{{/direct}}({{B}}[j*{{ldB}}+p]);
            }
            output += biases_{{name}}_{{idx}}[i];
        {{^fused_layer}}
            tensor_temp[i*{{ldC}}+j] = {{{activation_function}}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
            output = {{{activation_function}}};
            {{/linear}}
            tensor_temp[i*{{ldC}}+j] = {{{fused_layer}}};
        {{/fused_layer}}
        }
    }