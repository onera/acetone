    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; f++)
    {
        for (i = 0; i < {{output_height}}; i++)
        {
            for (j = 0; j < {{output_width}}; j++)
            {
                {{#concat}}
                    {{#channels}}
                if((f < {{borne_sup}}) && (f >= {{borne_inf}}))
                    tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[j + {{input_width}}*(i + {{input_height}}*(f - {{borne_inf}}))];
                    {{/channels}}
                    {{#heights}}
                if((i < {{borne_sup}}) && (i >= {{borne_inf}}))
                    tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[j + {{input_width}}*((i - {{borne_inf}}) + {{input_height}}*f)];
                    {{/heights}}
                    {{#widths}}
                if((j < {{borne_sup}}) && (j >= {{borne_inf}}))
                    tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = {{output_str}}[(j - {{borne_inf}}) + {{input_width}}*(i + {{input_height}}*f)];
                    {{/widths}}
                {{/concat}}
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_function}}};
    }