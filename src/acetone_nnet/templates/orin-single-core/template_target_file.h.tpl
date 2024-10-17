#ifndef __H_TARGET_H__
#define __H_TARGET_H__

#define MC 512
#define KC 256
#define NC 512

#define MR 4
#define NR 4

extern float packed_A[MC*KC];
extern float packed_B[KC*NC];

#define min(x,y)  (((x)<(y)) ? (x) : (y))

//
//  Packing panels from A with padding if required
//
void pack_A(int      mc, 
                   int      kc,  
                   float*   A, 
                   int      incRowA, 
                   int      incColA,
                   float*   buffer);

//
//  Packing panels from B with padding if required
//
void pack_B(int      kc, 
                   int      nc,  
                   float*   B, 
                   int      incRowB, 
                   int      incColB,
                   float*   buffer);

//
//  Micro kernel for multiplying panels from A and B.
//
void sgemm_micro_kernel(int      kc,
                               float*   A,  
                               float*   B,
                               float*   C, 
                               int      incRowC);

//
//  Macro kernel for multiplying panels from A and B.
//
void sgemm_macro_kernel(int      mc,
                               int      nc,
                               int      kc,
                               float*   C,
                               int      incRowC,
                               int      incColC);

#endif