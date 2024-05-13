    // gemm_tn
    for (i = 0; i < {{m}}; i++)
    {
        for(j = 0; j < {{n}}; ++j)
        {
            float register output = 0;
            for (p = 0; p < {{k}}; ++p)
            {
                output += {{#alpha}}{{.}}{{/alpha}} * {{A}}[i*{{k}}+p] * {{#direct}}*{{/direct}}({{B}}[p*{{n}}+j]);
            }   
            output += {{#beta}}{{.}}{{/beta}} * biases_{{name}}_{{idx}}[i];
        {{^fused_layer}}
            tensor_temp[j*{{m}} + i] = {{{activation_function}}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
            output = {{{activation_function}}};
            {{/linear}}
            tensor_temp[j*{{m}} + i] = {{{fused_layer}}};
        {{/fused_layer}}
            
        }
    }