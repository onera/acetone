#include "inference.h"

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
{{data_type}} *ppatches_{{name}}_{{idx}}[{{patches_size}}] = {{{patches}}};

    {{/patches_size}}
    {{#channels}}
{{data_type}} mean_{{name}}_{{idx}}[{{channels}}] = {{mean}};
{{data_type}} var_{{name}}_{{idx}}[{{channels}}] = {{var}};
{{data_type}} scale_{{name}}_{{idx}}[{{channels}}] = {{scale}};

    {{/channels}}
    {{#constant_size}}
{{data_type}} constant_{{name}}_{{idx}}[{{constant_size}}] = {{constant}};

    {{/constant_size}}
{{/layers}}