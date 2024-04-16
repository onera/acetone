#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

extern float weights_Dense_01[640];
extern float biases_Dense_01[128];
extern float weights_Dense_02[16384];
extern float biases_Dense_02[128];
extern float weights_Dense_03[8192];
extern float biases_Dense_03[64];
extern float weights_Dense_04[2048];
extern float biases_Dense_04[32];
extern float weights_Dense_05[512];
extern float biases_Dense_05[16];
extern float weights_Dense_06[80];
extern float biases_Dense_06[5];

int inference(float *prediction, float *nn_input);

#endif