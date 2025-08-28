    // gemm_tt
    for (i = 0; i < {{m}}; ++i)
    {
        for (j = 0; j < {{n}}; ++j)
        {
            float register sum = 0;
            for (p = 0; p < {{k}}; ++p)
            {
                sum += {{A}}[p*{{ldA}} +i]* {{#direct}}*{{/direct}}({{B}}[j*{{ldB}}+p]);
            }
            sum += biases_{{name}}_{{idx}}[i];
            tensor_temp[i*{{ldC}}+j] = {{{activation_function}}};
        }
    }