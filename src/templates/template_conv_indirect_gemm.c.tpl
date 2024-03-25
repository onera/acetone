    // {{name}}_{{idx}}
    {{#cst}}
    for (int k = 0; k < {{prev_size}}; ++k){
        tensor_temp[k] = output_{{road}}[k];
    }
    {{/cst}}
{{{gemm_code}}}
{{^channels_last}}
    for (int k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }
{{/channels_last}}
{{#channels_last}}
    for(int f = 0; f < {{input_channels}}; ++f)
    {
        for (int i = 0; i < {{input_height}}; ++i)
        {
            for (int j = 0; j < {{input_width}};  ++j)
            {
                output_{{road}}[(i*{{input_width}} + j)*{{input_channels}} + f] = tensor_temp[(f*{{input_height}} + i)*{{input_width}} + j];
            }
        }
    }
{{/channels_last}}
