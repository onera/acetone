    // gemm_nn
    for (i = 0; i < {{m}}; ++i)
    {
        for (p = 0; p < {{k}}; ++p)
        {
            register float weight = {{A}}[i*{{ldA}}+p];
            for(j = 0; j < {{n}}; ++j)
            {
                ctx->tensor_temp[i*{{ldC}} + j] += weight * {{#direct}}*{{/direct}}({{B}}[p*{{ldB}} + j]);
            }
        }
        for(j = 0; j < {{n}}; ++j)
        {
            register float output = ctx->tensor_temp[i*{{ldC}} + j];
            output += biases_{{name}}_{{idx}}[i];
            ctx->tensor_temp[i*{{ldC}} + j] = {{{activation_function}}};
        }
    }