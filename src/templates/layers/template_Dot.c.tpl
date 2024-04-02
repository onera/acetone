    // {{name}}_{{idx}}{{comment}}
    for (p = 0; p < {{output_fourth_dim}}; ++p)
    {
        for (f = 0; f < {{output_channels}}; ++f)
        {
            for (i = 0; i < {{output_height}}; ++i)
            {
                for (j = 0; j < {{output_width}}; ++j)
                {
                    register float output = 0;
                    for (k = 0; k < {{axis_dim}}; ++k)
                    {
                        output += {{output_str_left}}[{{indice_left}}] * {{output_str_right}}[{{indice_right}}];
                    }
                    output_{{road}}[j + {{output_width}}*(i + {{output_height}}*(f + {{output_channels}}*p))] = {{{activation_function}}};
                }
            }
        }
    }