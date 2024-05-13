    // gemm_tt
    for (i = 0; i < {{m}}; ++i)
    {
        for (j = 0; j < {{n}}; ++j)
        {
            float register sum = 0;
            for (p = 0; p < {{k}}; ++p)
            {
                sum += {{#alpha}}{{.}}{{/alpha}} * {{A}}[p*{{m}} +i]* {{#direct}}*{{/direct}}({{B}}[p*{{n}}+j]);
            }
            sum += {{#beta}}{{.}}{{/beta}} * biases_{{name}}_{{idx}}[i];
        {{^fused_layer}}
            tensor_temp[j*{{n}}+i] = {{{activation_function}}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
            sum = {{{activation_function}}};
            {{/linear}}
            tensor_temp[j*{{n}}+i] = {{{fused_layer}}};
        {{/fused_layer}}            
        }
    }