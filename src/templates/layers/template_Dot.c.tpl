    // {{name}}_{{idx}}{{comment}}
    for (int g = 0; g < {{output_fourth_dim}}; ++g)
    {
        for (int f = 0; f < {{output_channels}}; ++f)
        {
            for (int i = 0; i < {{output_height}}; ++i)
            {
                for (int j = 0; j < {{output_width}}; ++j)
                {
                    register float output = 0;
                    for (int k=0; k < {{axis_dim}}; ++k)
                    {
                        output += {{output_str_left}}[{{indice_left}}] * {{output_str_right}}[{{indice_right}}];
                    }
                    output_{{road}}[j + {{output_width}}*(i + {{output_height}}*(f + {{output_channels}}*g))] = {{{activation_function}}};
                }
            }
        }
    }