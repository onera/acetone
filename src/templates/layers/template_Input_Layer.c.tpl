    // {{name}}_{{idx}}
{{^channels_last}}
    // Charging the input in channels_first
    for (int i = 0; i < {{size}}; ++i)
    {
        output_{{road}}[i] = nn_input[i];
    }
{{/channels_last}}
{{#channels_last}}
    // Charging the input and changing it from channels_last to channels_first format
    for(int f = 0; f < {{input_channels}}; ++f)
    {
        for (int i = 0; i < {{input_height}}; ++i)
        {
            for (int j = 0; j < {{input_width}};  ++j)
            {
                output_{{road}}[(i + {{input_height}}*f)*{{input_width}} + j] = nn_input[(i*{{input_width}} + j)*{{input_channels}} + f];
            }
        }
    }
{{/channels_last}}