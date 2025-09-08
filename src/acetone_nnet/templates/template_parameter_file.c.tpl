#include "inference.h"
{{#zero}}
const {{data_type}} zero = 0.0f;
{{/zero}}

{{#layers}}
    {{#nb_weights}}
const {{data_type}} weights_{{name}}_{{idx}}[{{nb_weights}}] __attribute__((aligned({{page_size}}))) = {{weights}};

    {{/nb_weights}}
    {{#nb_biases}}
const {{data_type}} biases_{{name}}_{{idx}}[{{nb_biases}}]  __attribute__((aligned({{page_size}}))) = {{biases}};

    {{/nb_biases}}
    {{#patches_size}}
{{data_type}} *ppatches_{{name}}_{{idx}}[{{patches_size}}] __attribute__((aligned({{page_size}}))) = {{{patches}}};

    {{/patches_size}}
    {{#channels}}
const {{data_type}} mean_{{name}}_{{idx}}[{{channels}}] __attribute__((aligned({{page_size}}))) = {{mean}};
const {{data_type}} var_{{name}}_{{idx}}[{{channels}}] __attribute__((aligned({{page_size}}))) = {{var}};
const {{data_type}} scale_{{name}}_{{idx}}[{{channels}}] __attribute__((aligned({{page_size}}))) = {{scale}};

    {{/channels}}
    {{#constant_size}}
const {{data_type}} constant_{{name}}_{{idx}}[{{constant_size}}]  __attribute__((aligned({{page_size}}))) = {{constant}};

    {{/constant_size}}
{{/layers}}