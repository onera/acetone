#ifndef INFERENCE_H_ 
#define INFERENCE_H_ 

// output list for road 0
float output_pre_0[300];
float output_cur_0[300];



int inference(float *prediction, float *nn_input);

#endif