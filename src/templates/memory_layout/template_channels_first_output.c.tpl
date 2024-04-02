    // Returning the output in channels_first format (acetone code in channels_first)
    for (int k = 0; k < {{output_size}}; k++)
    {
        prediction[k] = output_{{road}}[k];
    }