#include "inference.h"

{{{synchro}}}

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

