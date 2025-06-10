    // gemm_nt
    for (j = 0; j < {{m}}; ++j)
    {
        for (i = 0; i < {{n}}; ++i)
        {
            register float output =0;
            for (p = 0; p < {{k}}; ++p)
            {
                output += {{#alpha}}{{.}}*{{/alpha}}{{A}}[p*{{n}}+i]*({{B}}[j*{{k}}+p]);
            }
            output += {{#beta}}{{.}}*{{/beta}}biases_{{name}}_{{idx}}[i];
        {{^fused_layer}}
            tensor_temp[j*{{n}}+i] = {{{activation_function}}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
            output = {{{activation_function}}};
            {{/linear}}
            tensor_temp[j*{{n}}+i] = {{{fused_layer}}};
        {{/fused_layer}}
        }
    }