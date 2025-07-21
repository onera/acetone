    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}

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
                tensor_temp[position] = {{{activation_function}}};
                position++;
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = tensor_temp[k];
    }