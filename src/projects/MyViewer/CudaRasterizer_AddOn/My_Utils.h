#pragma once


#define DEBUG_WATCH_CUDA_MEM(DST, SRC, TOTAL_SIZE) \
cudaMemcpy((void*)DST, SRC, TOTAL_SIZE, cudaMemcpyDeviceToHost);

// TEST