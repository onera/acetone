    // {{name}}_{{idx}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    // Load the constant
    {{#size}}
    for (k = 0; k < {{size}}; ++k)
    {
    output_{{road}}[k] = {{ weights_var }}[k];
    }
    {{/size}}