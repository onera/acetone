#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

// output list for road 0
float output_pre_0[600];
float output_cur_0[600];
// output list for road 1
float output_pre_1[600];
float output_cur_1[600];

extern float weights_Conv2D_02[81];
extern float biases_Conv2D_02[3];
extern float weights_Conv2D_01[81];
extern float biases_Conv2D_01[3];

int inference(float *prediction, float *nn_input);

#endif