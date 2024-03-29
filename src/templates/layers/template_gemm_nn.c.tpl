        // gemm_nn
    for (int i=0; i<{{m}}; i++){
        for (int p=0; p<{{k}}; ++p){
            register float weight = {{#alpha}}{{.}} * {{/alpha}}{{A}}[i*{{k}}+p];
            for(int j=0; j<{{n}}; ++j){
                tensor_temp[j*{{m}} + i] += weight * ({{B}}[j*{{k}} + p]);
            }
        }
        for(int j=0; j<{{n}}; ++j){
            register float output = tensor_temp[j*{{m}} + i];
            output += {{#beta}}{{.}} * {{/beta}}biases_{{name}}_{{idx}}[i];
        {{^fused_layer}}
            tensor_temp[j*{{m}} + i] = {{activation_function}};
        {{/fused_layer}}
        {{#fused_layer}}
            {{^linear}}
            output = {{activation_function}};
            {{/linear}}
            tensor_temp[j*{{m}} + i] = {{fused_layer}};
        {{/fused_layer}}
        }
    }