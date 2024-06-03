    // {{name}}_{{idx}}{{comment}}
    {{#all}}
    reduced = {{starting_value}};
    for (k = 0; k < {{size}}; ++k)
    {
        {{#Max}}
        if ({{output_str}}[k] > reduced)
        {
            reduced = {{output_str}}[k];
        }
        {{/Max}}
        {{#Min}}
        if ({{output_str}}[k] < reduced)
        {
            reduced = {{output_str}}[k];
        }
        {{/Min}}
        {{#Other}}
        reduced {{func}}= {{output_str}}[k];
        {{/Other}}
    }
    {{#Mean}}
    reduced = reduced/{{size}};
    {{/Mean}}

    output_{{road}}[0] = {{{activation_function}}};
    {{/all}}
    {{#two}}
    for (f = 0; f < {{output_dimension}}; ++f)
    {
        tensor_temp[f] = {{starting_value}};

        for (i = 0; i < {{reduced_dimension_1}}; ++i)
        {
            for (j = 0; j < {{reduced_dimension_2}}; ++j)
            {
                {{#Max}}
                if (output_{{road}}[{{position}}] > tensor_temp[f])
                {
                    tensor_temp[f] = {{output_str}}[{{position}}];
                }
                {{/Max}}
                {{#Min}}
                if (output_{{road}}[{{position}}] < tensor_temp[f])
                {
                    tensor_temp[f] = {{output_str}}[{{position}}];
                }
                {{/Min}}
                {{#Other}}
                tensor_temp[f] {{func}}= {{output_str}}[{{position}}];
                {{/Other}}
            }
        }
        {{#Mean}}
        tensor_temp[f] = tensor_temp[f]/{{nb_elements}};
        {{/Mean}}
    }

    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_function}}};
    }
    {{/two}}
    {{#one}}
    for (f = 0; f < {{output_dimension_1}}; ++f)
    {
        for (i = 0; i < {{output_dimension_2}}; ++i)
        {
            tensor_temp[{{position_1}}] = {{starting_value}};
            for (j = 0; j < {{reduced_dimension}}; ++j)
            {
                {{#Max}}
                if ({{output_str}}[{{position_2}}] > tensor_temp[{{position_1}}])
                {
                    tensor_temp[{{position_1}}] = {{output_str}}[{{position_2}}];
                }
                {{/Max}}
                {{#Min}}
                if ({{output_str}}[{{position_2}}] < tensor_temp[{{position_1}}])
                {
                    tensor_temp[{{position_1}}] = {{output_str}}[{{position_2}}];
                }
                {{/Min}}
                {{#Other}}
                tensor_temp[{{position_1}}] {{func}}= {{output_str}}[{{position_2}}];
                {{/Other}}
            }
            {{#Mean}}
            tensor_temp[{{position_1}}] = tensor_temp[{{position_1}}]/{{reduced_dimension}};
            {{/Mean}}
        }
    }
    
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_function}}};
    }
    {{/one}}
    {{#none}}
    //Act like a Linear layer
    {{#Activation}}
    for (k = 0; k < {{size}}; ++k)
    {
        output_{{road}}[k] = {{{activation_functions}}};
    }
    {{/Activation}}
    {{/none}}