    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    {{#cst}}
    for (k = 0; k < {{input_size}}; ++k)
    {
        tensor_temp[k] = output_{{road}}[k];
    }
    {{/cst}}
{{{patch_building_code}}}
    for (k = 0; k < {{patches_size}}; ++k)
    {
        tensor_temp[k] = 0;
    }
{{{gemm_code}}}
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }