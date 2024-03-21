#include "inference.h"

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
{{data_type}} weights_{{name}}_{{idx}}[{{nb_weights}}] = {{weights}};

    {{/nb_weights}}
    {{#nb_biases}}
{{data_type}} biases_{{name}}_{{idx}}[{{nb_biases}}] = {{biases}};

    {{/nb_biases}}
    {{#patches_size}}
{{data_type}} *ppatches_{{layers.name}}_{{layers.idx}}[{{patches_size}}] = {{patches}};

    {{/patches_size}}
{{/layers}}