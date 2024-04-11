    // post-processing for the normalization
    for (k = 0; k < {{output_size}}; ++k)
    {
        prediction[k] = prediction[k] * output_range + output_mean;
    }
