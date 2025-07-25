#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

{{#path}}
// output list for path {{.}}
extern {{data_type}} output_{{.}}[{{path_size}}];
{{/path}}

{{#cst}}
extern {{data_type}} cst_{{name}}[{{size}}];
{{/cst}}

{{#temp_size}}
extern {{temp_data_type}} tensor_temp[{{temp_size}}];

{{/temp_size}}
{{#layers}}
    {{#nb_weights}}
extern const {{data_type}} weights_{{name}}_{{idx}}[{{nb_weights}}];
    {{/nb_weights}}
    {{#nb_biases}}
extern const {{data_type}} biases_{{name}}_{{idx}}[{{nb_biases}}];
    {{/nb_biases}}
    {{#patches_size}}
extern {{data_type}} *ppatches_{{name}}_{{idx}}[{{patches_size}}];
    {{/patches_size}}
    {{#channels}}
extern const {{data_type}} mean_{{name}}_{{idx}}[{{channels}}];
extern const {{data_type}} var_{{name}}_{{idx}}[{{channels}}];
extern const {{data_type}} scale_{{name}}_{{idx}}[{{channels}}];
    {{/channels}}
    {{#constant_size}}
extern const {{data_type}} constant_{{name}}_{{idx}}[{{constant_size}}];
    {{/constant_size}}
    
{{/layers}}
int inference({{data_type}} *prediction, {{data_type}} *nn_input);

{{{normalization_cst}}}
#endif