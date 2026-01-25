    // gemm_tn
    for (i = 0; i < {{m}}; i++)
    {
        for(j = 0; j < {{n}}; ++j)
        {
            float register output = 0;
            for (p = 0; p < {{k}}; ++p)
            {
                output += {{#alpha}}{{.}}*{{/alpha}}{{A}}[p*{{m}}+i]*{{#direct}}*{{/direct}}({{B}}[p*{{n}}+j]);
            }   
            output += {{#beta}}{{.}}*{{/beta}}biases_{{name}}_{{idx}}[j];
            ctx->tensor_temp[i*{{n}} + j] = {{{activation_function}}};
        }
    }