#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

// output list for road 0
float output_0[788544];
// output list for road 1
float output_1[788544];

float cst_0[48400];

float tensor_temp[788544];

extern float weights_Conv2D_01[1728];
extern float biases_Conv2D_01[64];
extern float weights_Conv2D_03[1024];
extern float biases_Conv2D_03[16];
extern float weights_Conv2D_05[9216];
extern float biases_Conv2D_05[64];
extern float weights_Conv2D_04[1024];
extern float biases_Conv2D_04[64];
extern float weights_Conv2D_07[2048];
extern float biases_Conv2D_07[16];
extern float weights_Conv2D_09[9216];
extern float biases_Conv2D_09[64];
extern float weights_Conv2D_08[1024];
extern float biases_Conv2D_08[64];
extern float weights_Conv2D_12[4096];
extern float biases_Conv2D_12[32];
extern float weights_Conv2D_14[36864];
extern float biases_Conv2D_14[128];
extern float weights_Conv2D_13[4096];
extern float biases_Conv2D_13[128];
extern float weights_Conv2D_16[8192];
extern float biases_Conv2D_16[32];
extern float weights_Conv2D_18[36864];
extern float biases_Conv2D_18[128];
extern float weights_Conv2D_17[4096];
extern float biases_Conv2D_17[128];
extern float weights_Conv2D_21[12288];
extern float biases_Conv2D_21[48];
extern float weights_Conv2D_23[82944];
extern float biases_Conv2D_23[192];
extern float weights_Conv2D_22[9216];
extern float biases_Conv2D_22[192];
extern float weights_Conv2D_25[18432];
extern float biases_Conv2D_25[48];
extern float weights_Conv2D_27[82944];
extern float biases_Conv2D_27[192];
extern float weights_Conv2D_26[9216];
extern float biases_Conv2D_26[192];
extern float weights_Conv2D_29[24576];
extern float biases_Conv2D_29[64];
extern float weights_Conv2D_31[147456];
extern float biases_Conv2D_31[256];
extern float weights_Conv2D_30[16384];
extern float biases_Conv2D_30[256];
extern float weights_Conv2D_33[32768];
extern float biases_Conv2D_33[64];
extern float weights_Conv2D_35[147456];
extern float biases_Conv2D_35[256];
extern float weights_Conv2D_34[16384];
extern float biases_Conv2D_34[256];
extern float weights_Conv2D_37[512000];
extern float biases_Conv2D_37[1000];

int inference(float *prediction, float *nn_input);

#endif