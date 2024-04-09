    // Returning the output in channels_first format (acetone code in channels_first)
    for (k = 0; k < {{output_size}}; ++k)
    {
        prediction[k] = output_{{path}}[k];
    }