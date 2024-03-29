    // {{name}}_{{idx}}
    for (int k = 0; k < {{patches_size}}; ++k)
    {
        tensor_temp[k] = 0;
    }
{{{gemm_code}}}
    for (int k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }