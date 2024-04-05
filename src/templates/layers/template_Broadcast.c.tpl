    // {{name}}_{{idx}}{{comment}}
    for (f = 0; f < {{output_channels}}; ++f)
    {
        for (i = 0; i < {{output_height}}; ++i)
        {
            for (j = 0; j < {{output_width}}; ++j)
            {
                {{#max}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = max({{#broadcast}}{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]{{operator}}{{/broadcast}});
                {{/max}}
                {{#min}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = min({{#broadcast}}{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]{{operator}}{{/broadcast}});
                {{/min}}
                {{#Average}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] = ({{#broadcast}}{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]{{operator}}{{/broadcast}})/{{prev_size}};
                {{/Average}}
                {{#other}}
                tensor_temp[j + {{output_width}}*(i + {{output_height}}*f)] ={{#broadcast}}{{output_str}}[(j%{{input_width}}) + {{input_width}}*((i%{{input_height}}) + {{input_height}}*(f % {{input_channels}}))]{{operator}}{{/broadcast}};
                {{/other}}
            }
        }
    }
    
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_function}}};
    }