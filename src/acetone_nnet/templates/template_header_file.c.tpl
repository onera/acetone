#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

typedef struct inference_t{
{{#path}}
// output list for path {{.}}
    {{data_type}} output_{{.}}[{{path_size}}] __attribute__((aligned({{page_size}})));
{{/path}}
{{#cst}}
    {{data_type}} cst_{{name}}[{{size}}] __attribute__((aligned({{page_size}})));
{{/cst}}
{{#temp_size}}
    {{temp_data_type}} tensor_temp[{{temp_size}}] __attribute__((aligned({{page_size}})));
{{/temp_size}}
}inference_t;

#define MAX_BATCH_SIZE {{max_batch_size}}
/* Activation and temp tensor allocation */
extern inference_t Context[MAX_BATCH_SIZE];

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
int inference(inference_t *context, {{data_type}} *prediction, {{data_type}} *nn_input);

{{{normalization_cst}}}
#endif