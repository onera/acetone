    // Returning the output in channels_last format (changing from channels_first to channels_last)
    for(int f = 0; f < {{output_channels}}; ++f)
    {
        for (int i = 0; i < {{output_height}}; ++i)
        {
            for (int j = 0; j < {{output_width}};  ++j)
            {
                prediction[(i*{{output_width}} + j)*{{output_channels}} + f] = output_{{road}}[(f*{{output_height}} + i)*{{output_width}} + j];
            }
        }
    }