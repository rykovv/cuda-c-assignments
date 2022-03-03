#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include <wb.h>

#define THREADS_PER_BLOCK 128

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

__host__ void errchk(cudaError_t error_code, int line) {
    if (error_code) {
        printf("\"%s\" in %s at line %d\n",
            cudaGetErrorString(error_code), __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  errchk(cudaMalloc((void**)&deviceInput1, inputLength * sizeof(float)), __LINE__);
  errchk(cudaMalloc((void**)&deviceInput2, inputLength * sizeof(float)), __LINE__);
  errchk(cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float)), __LINE__);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  errchk(cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice), __LINE__);
  errchk(cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice), __LINE__);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((inputLength - 1) / THREADS_PER_BLOCK +1, 1, 1);
  dim3 DimBlock(THREADS_PER_BLOCK, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd <<<DimGrid, DimBlock>>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
  errchk(cudaDeviceSynchronize(), __LINE__);
  wbTime_stop(Compute, "Performing CUDA computation");
 
  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  errchk(cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  errchk(cudaFree(deviceInput1), __LINE__);
  errchk(cudaFree(deviceInput2), __LINE__);
  errchk(cudaFree(deviceOutput), __LINE__);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
