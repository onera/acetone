    //{{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    for(f = 0; f < {{input_channels}}; ++f)
    {
        for(k = 0; k < {{channel_size}}; ++k)
        {
            ctx->output_{{path}}[k + {{channel_size}}*f] = scale_{{name}}_{{idx}}[f]*(ctx->{{output_str}}[k + {{channel_size}}*f] - mean_{{name}}_{{idx}}[f])/sqrt(var_{{name}}_{{idx}}[f] + {{epsilon}}) + biases_{{name}}_{{idx}}[f];
            {{#activation_function}}
            ctx->output_{{path}}[k + {{channel_size}}*f] = {{{activation_function}}};
            {{/activation_function}}
        }
    }