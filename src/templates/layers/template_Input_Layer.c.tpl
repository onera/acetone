    // {{name}}_{{idx}}
{{^channels_last}}
    // Charging the input in channels_first
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = nn_input[k];
    }
{{/channels_last}}
{{#channels_last}}
    // Charging the input and changing it from channels_last to channels_first format
    for(f = 0; f < {{input_channels}}; ++f)
    {
        for (i = 0; i < {{input_height}}; ++i)
        {
            for (j = 0; j < {{input_width}};  ++j)
            {
                output_{{road}}[(i + {{input_height}}*f)*{{input_width}} + j] = nn_input[(i*{{input_width}} + j)*{{input_channels}} + f];
            }
        }
    }
{{/channels_last}}