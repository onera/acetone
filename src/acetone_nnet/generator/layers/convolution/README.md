# Convolution Algorithm
  
From the slowest to the fastest  

## Direct Convolution  
class Conv2D6loops  
versions={"Conv2D": "6loops"}  

## Im2Col+Gemm
class Conv2DStdGemm  

t means transposed  
versions={"Conv2D": "std_gemm_nn"}  
versions={"Conv2D": "std_gemm_tn"}  
versions={"Conv2D": "std_gemm_nt"}  
versions={"Conv2D": "std_gemm_tt"}  

## Direct Convolution with channel blocks
class Conv2Ddirect_block  
parameters are input channel blocks and output channel blocs.  
On AVX512, best output channel is 64, input channel does not improve (thks to big cache).
Shall be evaluated on target HW with smaller caches.

versions={"Conv2D": "direct_block"}
