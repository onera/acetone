    // {{name}}_{{idx}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    // Wait for the communication tensor to be ready then write in it and notify the destination core
    while(synchro_{{current_core}}_{{dst_core}} != 0);
    for (k = 0; k < {{size}}; ++k)
    {
        com_{{current_core}}_{{dst_core}}[k] = {{output_str}}[k];
    }
    *synchro_{{current_core}}_{{dst_core}} = 1;