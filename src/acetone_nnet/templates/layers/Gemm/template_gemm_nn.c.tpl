        // gemm_nn
    for (i = 0; i < {{m}}; ++i)
    {
        for (p = 0; p < {{k}}; ++p)
        {
            register float weight = {{#alpha}}{{.}}*{{/alpha}}{{A}}[i*{{k}}+p];
            for(j = 0; j < {{n}}; ++j)
            {
                tensor_temp[j*{{m}} + i] += weight*({{B}}[j*{{k}} + p]);
            }
        }
        for(j = 0; j<{{n}}; ++j)
        {
            register float output = tensor_temp[j*{{m}} + i];
            output += {{#beta}}{{.}}*{{/beta}}biases_{{name}}_{{idx}}[i];
            tensor_temp[j*{{m}} + i] = {{{activation_function}}};
        }
    }