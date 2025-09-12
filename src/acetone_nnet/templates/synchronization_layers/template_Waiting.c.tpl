    // {{name}}_{{idx}} {{#original_name}}(layer {{original_name}} in  input model){{/original_name}}
    // Wait for to read a communication
    while(synchro_{{src_core}}_{{current_core}} != 1);