    // {{name}}_{{idx}}{{comment}}

    position = 0;

    {{#channels}}
    for (int k = 0; k < {{indices_len}}; ++k)
    {
        int f = indice_Gather_{{idx}}[k];
        for (int i = 0; i < {{output_height}}; ++i)
        {
            for (int j = 0; j < {{output_width}}; ++j)
            {
    {{/channels}}
    {{#heights}}
    for (int f = 0; f < {{output_channels}}; ++f)
    {
        for (int k = 0; k < {{indices_len}}; ++k)
        {
            int i = indice_Gather_{{idx}}[k];
            for (int j = 0; j < {{output_width}}; ++j)
            {
    {{/heights}}
    {{#widths}}
    for (int f = 0; f < {{output_channels}}; ++f)
    {
        for (int i = 0; i < {{output_height}}; ++i)
        {
            for (int k = 0; k < {{indices_len}}; ++k)
            {
                int j = indice_Gather_{{idx}}[k];
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
    for (int k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }