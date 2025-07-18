    // gemm_nt
    for (i = 0; i < {{m}}; ++i)
    {
        for (j = 0; j < {{n}}; ++j)
        {
            register float output =0;
            for (p = 0; p < {{k}}; ++p)
            {
                output += {{#alpha}}{{.}}*{{/alpha}}{{A}}[p*{{m}}+i]*({{B}}[j*{{k}}+p]);
            }
            output += {{#beta}}{{.}}*{{/beta}}biases_{{name}}_{{idx}}[i];
            tensor_temp[j*{{m}}+i] = {{{activation_function}}};
        }
    }