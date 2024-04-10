#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

// output list for road 0
float output_pre_0[500];
float output_cur_0[500];


extern float weights_Dense_01[125000];
extern float biases_Dense_01[250];

int inference(float *prediction, float *nn_input);

#endif