#include "inference.h"

{{#road}}
// output list for road {{.}}
{{data_type}} output_{{.}}[{{road_size}}];
{{/road}}

{{#constant}}
{{data_type}} cst_{{constant.name}}[{{constant.size}}];
{{/constant}}

{{#tensor_temp}}
{{data_type}} tensor_temp[{{temp_size}}];
{{/tensor_temp}}

{{#zero}}
{{data_type}} zero = 0.0f;
{{/zero}}

