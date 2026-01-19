{{#keep_channels}}
    // Output is in same data_format as Input
    for (k = 0; k < {{output_size}}; ++k)
    {
        prediction[k] = output_{{path}}[k];
    }
{{/keep_channels}}
{{#channels_first_to_last}}
    // Convert output layer from channels_first to channels_last
    for(f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}};  ++j)
            {
                prediction[(i*{{output_width}} + j)*{{output_channels}} + f] = output_{{path}}[(f*{{output_height}} + i)*{{output_width}} + j];
            }
        }
    }
{{/channels_first_to_last}}
{{#channels_last_to_first}}
    // Convert output layer from channels_last to channels_first
    for(f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}};  ++j)
            {
                prediction[(f*{{output_height}} + i)*{{output_width}} + j] = output_{{path}}[(i*{{output_width}} + j)*{{output_channels}} + f];
            }
        }
    }    
{{/channels_last_to_first}}
