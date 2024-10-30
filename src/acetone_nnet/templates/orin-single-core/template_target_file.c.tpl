#include "target.h"

float packed_A[MC*KC] __attribute__((__aligned__(64)));
float packed_B[KC*NC] __attribute__((__aligned__(64)));

//
//  Packing panels from A with padding if required
//
void pack_A(int      mc, 
                   int      kc,  
                   float*   A, 
                   int      incRowA, 
                   int      incColA,
                   float*   buffer)
{
    int register mp  = mc / MR;
    int register _mr = mc % MR;

    int register i, j, l;
    register float* _A = A;
    register float* _buffer = buffer;
    int register _incRowA = incRowA, _incColA = incColA, _kc = kc;

    for (l=0; l<mp; ++l) {
        for (j=0; j<_kc; ++j) {
            for (i=0; i<MR; ++i) {
                _buffer[i] = _A[i*_incRowA+j*_incColA];
            }
            _buffer += MR;
        }
        _A += MR*_incRowA;
    }
    if (_mr>0) {
        for (j=0; j<_kc; ++j) {
            for (i=0; i<_mr; ++i) {
                _buffer[i] = _A[i*_incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                _buffer[i] = 0.0;
            }
            _buffer += MR;
            _A += _incColA;
        }
    }
}

//
//  Packing panels from B with padding if required
//
void pack_B(int      kc, 
                   int      nc,  
                   float*   B, 
                   int      incRowB, 
                   int      incColB,
                   float*   buffer)
{
    int register np  = nc / NR;
    int register _nr = nc % NR;

    int register i, j, l;
    register float* _B = B;
    register float* _buffer = buffer;
    int register _incRowB = incRowB, _incColB = incColB, _kc = kc;

    for (l=0; l<np; ++l) {
        for (i=0; i<_kc; ++i) {
            for (j=0; j<NR; ++j) {
                _buffer[j] = _B[j*_incColB + i*_incRowB];
            }
            _buffer += NR;
        }
        _B += NR*_incColB;
    }

    if (_nr>0) {
        for (i=0; i<_kc; ++i) {
            for (j=0; j<_nr; ++j) {
                _buffer[j] = _B[j*_incColB];
            }
            for (j=_nr; j<NR; ++j) {
                _buffer[j] = 0.0;
            }
            _buffer += NR;
            _B      += _incRowB;
        }
    }
}

//
//  Micro kernel for multiplying panels from A and B.
//
void sgemm_micro_kernel(int      kc,
                               float*   A,  
                               float*   B,
                               float*   C, 
                               int      incRowC)
{
    register float* _C asm ("x4") = C;
    register float* _A asm ("x6") = A;
    register float* _B asm ("x7") = B;
    register int _incRowC asm ("x5") = incRowC;

    register int _kc asm ("x1") = kc;
    register int l   asm ("x2") = 0;

    __asm__  (
        "MOV w3, #0                         \n\t"
        "DUP V0.4s, w3                      \n\t" // duplicate 0.0 to V0 
        "DUP V1.4s, w3                      \n\t" // duplicate 0.0 to V1 
        "DUP V2.4s, w3                      \n\t" // duplicate 0.0 to V2 
        "DUP V3.4s, w3                      \n\t" // duplicate 0.0 to V3 

        // Load the C rows as column

        "LD1 {v0.4s}, [x4]                  \n\t" // Load 1rst row of C in v0   

        "ADD x3, x4, x5, lsl #2             \n\t" // x3 = _C (x4) + 1*incRowC*sizeof(float) (x5*4)
        "LD1 {v1.4s}, [x3]                  \n\t" // Load  2nd row of C in v1

        "ADD x3, x3, x5, lsl #2             \n\t" // x3 = _C (x4) + 2*incRowC*sizeof(float) 
        "LD1 {v2.4s}, [x3]                  \n\t" // Load  3rd row of C in v2

        "ADD x3, x3, x5, lsl #2             \n\t" // x3 = _C (x4) + 3*incRowC*sizeof(float) 
        "LD1 {v3.4s}, [x3]                  \n\t" // Load  4th row of C in v3

        // begin computation
        
        "loop:                              \n\t"
        "LD1 {v4.4s}, [x6]                  \n\t"
        "LD1 {v5.4s}, [x7]                  \n\t"

        "FMLA v0.4s, v5.4s, v4.s[0]         \n\t"
        "FMLA v1.4s, v5.4s, v4.s[1]         \n\t"
        "FMLA v2.4s, v5.4s, v4.s[2]         \n\t"
        "FMLA v3.4s, v5.4s, v4.s[3]         \n\t"

        "ADD x6, x6, #16                    \n\t" // A + 4*sizeof(float);
        "ADD x7, x7, #16                    \n\t" // B + 4*sizeof(float);
        "ADD x2, x2, #1                     \n\t"
        "CMP x2, x1                         \n\t"
        "B.LT loop                          \n\t"

        // store results
        
        "ST1 {v0.4s}, [x4]                  \n\t" // store 1rst row of C

        "ADD x3, x4, x5, lsl #2             \n\t" // x3 = _C (x4) + 1*incRowC*sizeof(float) (x5*4)
        "ST1 {v1.4s}, [x3]                  \n\t" // store 2nd row of C

        "ADD x3, x3, x5, lsl #2             \n\t" // x3 = _C (x4) + 1*incRowC*sizeof(float) (x5*4)
        "ST1 {v2.4s}, [x3]                  \n\t" // store 3rd row of C

        "ADD x3, x3, x5, lsl #2             \n\t" // x3 = _C (x4) + 1*incRowC*sizeof(float) (x5*4)
        "ST1 {v3.4s}, [x3]                  \n\t" // store 4th row of C
        : // outputs operands
        :  
        :  "x3", "v0", "v1", "v2", "v3", "v4", "v5"          // register clobber list
    );
}

//
//  Macro kernel for multiplying panels from A and B.
//
void sgemm_macro_kernel(int      mc,
                               int      nc,
                               int      kc,
                               float*   C,
                               int      incRowC,
                               int      incColC)
{
    int register _mc = mc;
    int register _nc = nc;
    int register _kc = kc;

    int register _incRowC = incRowC;
    int register _incColC = incColC;

    int register _i, _j;

    for (_j = 0; _j < _nc; _j += NR ) {
        for (_i = 0; _i < _mc; _i += MR ) {
            sgemm_micro_kernel(_kc, &packed_A[_i*_kc], &packed_B[_j*_kc], &C[_i*_incRowC+_j*_incColC], _incRowC);
        }
    }
}