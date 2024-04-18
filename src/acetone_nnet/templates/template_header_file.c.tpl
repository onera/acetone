#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

{{#path}}
// output list for path {{.}}
{{data_type}} output_{{.}}[{{path_size}}];
{{/path}}

{{#cst}}
    {{#name}}
{{data_type}} cst_{{name}}[{{size}}];
    {{/name}}
{{/cst}}

{{#tensor_temp}}
{{data_type}} tensor_temp[{{temp_size}}];

{{/tensor_temp}}
{{#layers}}
    {{#nb_weights}}
extern {{data_type}} weights_{{name}}_{{idx}}[{{nb_weights}}];
    {{/nb_weights}}
    {{#nb_biases}}
extern {{data_type}} biases_{{name}}_{{idx}}[{{nb_biases}}];
    {{/nb_biases}}
    {{#patches_size}}
extern {{data_type}} *ppatches_{{name}}_{{idx}}[{{patches_size}}];
    {{/patches_size}}
    {{#channels}}
extern {{data_type}} mean_{{name}}_{{idx}}[{{channels}}];
extern {{data_type}} var_{{name}}_{{idx}}[{{channels}}];
extern {{data_type}} scale_{{name}}_{{idx}}[{{channels}}];
    {{/channels}}
    {{#constant_size}}
extern {{data_type}} constant_{{name}}_{{idx}}[{{constant_size}}];
    {{/constant_size}}
    
{{/layers}}
int inference({{data_type}} *prediction, {{data_type}} *nn_input);

{{{normalization_cst}}}
#endif