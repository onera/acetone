#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

{{#road}}
// output list for road {{.}}
{{data_type}} output_{{.}}[{{road_size}}];
{{/road}}

{{#cst}}
    {{#name}}
{{data_type}} cst_{{name}}[{{size}}];
    {{/name}}
{{/cst}}

{{#tensor_temp}}
{{data_type}} tensor_temp[{{temp_size}}];

{{/tensor_temp}}
{{#zero}}
{{data_type}} zero = 0.0f;

{{/zero}}
{{#layers}}
    {{#nb_weights}}
extern {{data_type}} weights_{{name}}_{{idx}}[{{nb_weights}}];

    {{/nb_weights}}
    {{#nb_biases}}
extern {{data_type}} biases_{{name}}_{{idx}}[{{nb_biases}}];

    {{/nb_biases}}
    {{#patches_size}}
extern {{data_type}} *ppatches_{{layers.name}}_{{layers.idx}}[{{patches_size}}];

    {{/patches_size}}
{{/layers}}
int inference({{data_type}} *prediction, {{data_type}} *nn_input);

#endif