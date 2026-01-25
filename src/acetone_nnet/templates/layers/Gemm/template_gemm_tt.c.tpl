    // gemm_tt
    for (i = 0; i < {{m}}; ++i)
    {
        for (j = 0; j < {{n}}; ++j)
        {
            float register sum = 0;
            for (p = 0; p < {{k}}; ++p)
            {
                sum += {{#alpha}}{{.}}*{{/alpha}}{{A}}[p*{{m}} +i]*{{#direct}}*{{/direct}}({{B}}[j*{{k}}+p]);
            }
            sum += {{#beta}}{{.}}*{{/beta}}biases_{{name}}_{{idx}}[j];
            ctx->tensor_temp[i*{{n}}+j] = {{{activation_function}}};
        }
    }