    // {{name}}_{{idx}}
{{{gemm_code}}}
    for (int k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }
