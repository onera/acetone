    // {{name}}_{{idx}}{{comment}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    {{#all}}
    reduced = {{{starting_value}}};
    for (k = 0; k < {{size}}; ++k)
    {
        {{#Max}}
        if (ctx->{{output_str}}[k] > reduced)
        {
            reduced = ctx->{{output_str}}[k];
        }
        {{/Max}}
        {{#Min}}
        if (ctx->{{output_str}}[k] < reduced)
        {
            reduced = ctx->{{output_str}}[k];
        }
        {{/Min}}
        {{#Other}}
        reduced {{func}}= ctx->{{output_str}}[k];
        {{/Other}}
    }
    {{#Mean}}
    reduced = reduced/{{size}};
    {{/Mean}}

    ctx->output_{{road}}[0] = {{{activation_function}}};
    {{/all}}
    {{#two}}
    for (f = 0; f < {{output_dimension}}; ++f)
    {
        ctx->tensor_temp[f] = {{{starting_value}}};

        for (i = 0; i < {{reduced_dimension_1}}; ++i)
        {
            for (j = 0; j < {{reduced_dimension_2}}; ++j)
            {
                {{#Max}}
                if (ctx->output_{{road}}[{{position}}] > ctx->tensor_temp[f])
                {
                    ctx->tensor_temp[f] = ctx->{{output_str}}[{{position}}];
                }
                {{/Max}}
                {{#Min}}
                if (ctx->output_{{road}}[{{position}}] < ctx->tensor_temp[f])
                {
                    ctx->tensor_temp[f] = ctx->{{output_str}}[{{position}}];
                }
                {{/Min}}
                {{#Other}}
                ctx->tensor_temp[f] {{func}}= ctx->{{output_str}}[{{position}}];
                {{/Other}}
            }
        }
        {{#Mean}}
        ctx->tensor_temp[f] = ctx->tensor_temp[f]/{{nb_elements}};
        {{/Mean}}
    }

    for (k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = {{{activation_function}}};
    }
    {{/two}}
    {{#one}}
    for (f = 0; f < {{output_dimension_1}}; ++f)
    {
        for (i = 0; i < {{output_dimension_2}}; ++i)
        {
            ctx->tensor_temp[{{position_1}}] = {{{starting_value}}};
            for (j = 0; j < {{reduced_dimension}}; ++j)
            {
                {{#Max}}
                if (ctx->{{output_str}}[{{position_2}}] > ctx->tensor_temp[{{position_1}}])
                {
                    ctx->tensor_temp[{{position_1}}] = ctx->{{output_str}}[{{position_2}}];
                }
                {{/Max}}
                {{#Min}}
                if (ctx->{{output_str}}[{{position_2}}] < ctx->tensor_temp[{{position_1}}])
                {
                    ctx->tensor_temp[{{position_1}}] = ctx->{{output_str}}[{{position_2}}];
                }
                {{/Min}}
                {{#Other}}
                ctx->tensor_temp[{{position_1}}] {{func}}= ctx->{{output_str}}[{{position_2}}];
                {{/Other}}
            }
            {{#Mean}}
            ctx->tensor_temp[{{position_1}}] = ctx->tensor_temp[{{position_1}}]/{{reduced_dimension}};
            {{/Mean}}
        }
    }
    
    for (k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = {{{activation_function}}};
    }
    {{/one}}
    {{#none}}
    //Act like a Linear layer
    for (k = 0; k < {{size}}; ++k)
    {
        ctx->output_{{road}}[k] = {{{activation_function}}};
    }
    {{/none}}