{{#channels_last}}
    //{{name}}_{{idx}}  {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    // To transforme the tensor in channels_last before entering the 1D layers
    for(f = 0; f < {{input_channels}}; ++f)
    {
        for (i = 0; i < {{input_height}}; ++i)
        {
            for (j = 0; j < {{input_width}};  ++j)
            {
                ctx->tensor_temp[(i*{{input_width}} + j)*{{input_channels}} + f] = output_{{path}}[(f*{{input_height}} + i)*{{input_width}} + j];
            }
        }
    }
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{path}}[k] = ctx->tensor_temp[k];
    }
{{/channels_last}}