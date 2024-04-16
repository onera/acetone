#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

extern float weights_Conv2D_01[150];
extern float biases_Conv2D_01[6];
extern float weights_Conv2D_03[2400];
extern float biases_Conv2D_03[16];
extern float weights_Dense_05[30720];
extern float biases_Dense_05[120];
extern float weights_Dense_06[10080];
extern float biases_Dense_06[84];
extern float weights_Dense_07[840];
extern float biases_Dense_07[10];

int inference(float *prediction, float *nn_input);

#endif