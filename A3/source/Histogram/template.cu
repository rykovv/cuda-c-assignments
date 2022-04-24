#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define NUM_BINS    4096
#define BLOCK_SIZE  512
#define SAT_MAX     127

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
    bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
        if (abort)
            exit(code);
    }
}

__global__ void histogram(unsigned int* input, unsigned int* bins,
    unsigned int num_elements,
    unsigned int num_bins) {
    //@@ Write the kernel that computes the histogram
    //@@ Make sure to use the privitization technique
    //(hint: since NUM_BINS=4096 is larger than maximum allowed number of threads per block,
    //be aware that threads would need to initialize more than one shared memory bin
    //and update more than one global memory bin)
    __shared__ unsigned int private_bins[NUM_BINS];

    if (threadIdx.x < NUM_BINS) {
        private_bins[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int nbin = 0;

    while (i < num_elements) {
        nbin = input[i] / ((num_elements - 1) / num_bins + 1);
        atomicAdd(&(private_bins[nbin]), 1);
        i += stride;
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&(bins[threadIdx.x]), private_bins[threadIdx.x]);
    }
}

__global__ void saturate(unsigned int* bins, unsigned int num_bins) {
    //@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_bins) {
        bins[i] = min(bins[i], SAT_MAX);
    }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int inputLength;
    unsigned int* hostInput;
    unsigned int* hostBins;
    unsigned int* deviceInput;
    unsigned int* deviceBins;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (unsigned int*)wbImport(wbArg_getInputFile(args, 0),
        &inputLength, "Integer");
    hostBins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);
    wbLog(TRACE, "The number of bins is ", NUM_BINS);

    wbTime_start(GPU, "Allocating device memory");
    CUDA_CHECK(cudaMalloc((void**)&deviceInput, inputLength * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&deviceBins, NUM_BINS * sizeof(unsigned int)));
    //@@ Allocate device memory here
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating device memory");

    wbTime_start(GPU, "Copying input host memory to device");
    //@@ Copy input host memory to device
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Copying input host memory to device");

    wbTime_start(GPU, "Clearing the bins on device");
    //@@ zero out the deviceBins using cudaMemset()
    CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
    wbTime_stop(GPU, "Clearing the bins on device");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((inputLength - 1) / BLOCK_SIZE + 1, 1, 1);
    //dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbLog(TRACE, "Launching kernel");
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Invoke kernels: first call histogram kernel and then call saturate kernel
    histogram << < DimGrid, DimBlock >> > (
        deviceInput,
        deviceBins,
        inputLength,
        NUM_BINS
        );

    saturate << < DimGrid, DimBlock >> > (deviceBins, NUM_BINS);
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output device memory to host");
    //@@ Copy output device memory to host
    CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(Copy, "Copying output device memory to host");

    wbTime_start(GPU, "Freeing device memory");
    //@@ Free the device memory here
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceBins));
    wbTime_stop(GPU, "Freeing device memory");

    wbSolution(args, hostBins, NUM_BINS);

    free(hostBins);
    free(hostInput);
    return 0;
}
