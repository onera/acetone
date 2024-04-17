    //{{name}}_{{idx}}{{comment}}
    for(f = 0; f < {{input_channels}}; ++f)
    {
        for(k = 0; k < {{channel_size}}; ++k)
        {
            output_{{path}}[k + {{channel_size}}*f] = scale_{{name}}_{{idx}}[f]*({{output_str}}[k + {{channel_size}}*f] - mean_{{name}}_{{idx}}[f])/sqrt(var_{{name}}_{{idx}}[f] + {{epsilon}}) + biases_{{name}}_{{idx}}[f];
            {{#activation_function}}
            output_{{path}}[k + {{channel_size}}*f] = {{{activation_function}}};
            {{/activation_function}}
        }
    }