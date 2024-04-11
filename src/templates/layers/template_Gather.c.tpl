    // {{name}}_{{idx}}{{comment}}

    position = 0;

    {{#channels}}
    for (k = 0; k < {{indices_len}}; ++k)
    {
        f = indice_Gather_{{idx}}[k];
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
    {{/channels}}
    {{#heights}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (k = 0; k < {{indices_len}}; ++k)
        {
            i = indice_Gather_{{idx}}[k];
            for (j = 0; j < {{output_width}}; ++j)
            {
    {{/heights}}
    {{#widths}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (k = 0; k < {{indices_len}}; ++k)
            {
                j = indice_Gather_{{idx}}[k];
    {{/widths}}
                tensor_temp[position] = {{output_str}}[j + {{input_width}}*(i + {{input_height}}*f)];
            {{^linear}}
                tensor_temp[position] = {{{activation_function}}};
            {{/linear}}
            {{#fused_layer}}
                tensor_temp[position] = {{{fused_layer}}};
            {{/fused_layer}}
                position++;
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }