    // pre-processing for the normalization
    for (i = 0; i < {{input_size}}; ++i) 
    { 
        if ( nn_input[i] < input_min[i]){
            nn_input[i] = (input_min[i]-input_mean[i])/input_range[i];
        }
        else
        {
            if (nn_input[i] > input_max[i]){
                nn_input[i] = (input_max[i]-input_mean[i])/input_range[i];
            }
            else
            {
                nn_input[i] = (nn_input[i]-input_mean[i])/input_range[i];
            }
        }
    }
    