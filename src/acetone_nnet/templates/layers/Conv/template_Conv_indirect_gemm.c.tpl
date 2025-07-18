    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for (k = 0; k < {{size}}; ++k)
    {
        tensor_temp[k] = 0;
    }
{{{gemm_code}}}
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }
