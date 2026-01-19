    // {{name}}_{{idx}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
{{#keep_channels}}
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = nn_input[k];
    }
{{/keep_channels}}
{{#channels_first_to_last}}
    // Loading the input and channels first to channels last
    for(f = 0; f < {{input_channels}}; ++f)
    {
        for (i = 0; i < {{input_height}}; ++i)
        {
            for (j = 0; j < {{input_width}};  ++j)
            {
                output_{{road}}[(i*{{input_width}} + j)*{{input_channels}} + f] = nn_input[(i + {{input_height}}*f)*{{input_width}} + j];
            }
        }
    }
{{/channels_first_to_last}}
{{#channels_last_to_first}}
    // Loading the input and channels last to channels first
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
{{/channels_last_to_first}}
