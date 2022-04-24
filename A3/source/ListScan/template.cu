#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define BLOCK_SIZE 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__host__ unsigned nblocks_scan(unsigned len) {
    return ((len - 1) / (2 * BLOCK_SIZE) + 1);
}

__global__ void scan(float* input, float* output, float* aux, int len) {
    //@@ Modify the body of this kernel to generate the scanned blocks
    //@@ Make sure to use the workefficient version of the parallel scan
    //@@ Also make sure to store the block sum to the aux array

    // Data loading
    __shared__ float XY[2 * BLOCK_SIZE];

    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        XY[threadIdx.x] = input[i];
    }
    if (i + blockDim.x < len) {
        XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
    }

    // Reduction phase
    for (unsigned stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        // once the shared memory is populated or intermediate computation is done
        __syncthreads();
        // legacy indexing
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    // Post reduction reverse phase
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        // once previous computations are done
        __syncthreads();
        // legacy index
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x) {
            XY[index + stride] += XY[index];
        }
    }
    // finish all the computations. Do we really need it? Yes! Dataset 2 fails without it.
    __syncthreads();
    // store outputs
    if (i < len) {
        output[i] = XY[threadIdx.x];
    }
    if (i + blockDim.x < len) {
        output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }

    if (aux) {
        if (i == len - 1) {
            aux[blockIdx.x] = XY[threadIdx.x];
        }
        else if (threadIdx.x == blockDim.x - 1) {
            // what is faster BLOCK_SIZE comparisons + 1 store or BLOCK_SIZE stores?
            aux[blockIdx.x] = XY[2 * blockDim.x - 1];
        }
    }
}

__global__ void addScannedBlockSums(float* output, const float* __restrict__ aux, int len) {
    //@@ Modify the body of this kernel to add scanned block sums to
    //@@ all values of the scanned blocks
    int i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    if (i < len) {
        output[i] += aux[blockIdx.x];
    }
}

int main(int argc, char** argv) {
    wbArg_t args;
    float* hostInput;  // The input 1D list
    float* hostOutput; // The output 1D list
    float* deviceInput;
    float* deviceOutput;
    float* deviceAuxArray, * deviceAuxScannedArray;
    int numElements; // number of elements in the input/output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

    wbTime_start(GPU, "Allocating device memory.");
    //@@ Allocate device memory
    wbCheck(cudaMalloc((void**)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements * sizeof(float)));
    //you can assume that aux array size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
    // In fact, aux arrays' length will be equal to the number of blocks.
    // I will keep the assumption that the number of blocks will be less than 2*BLOCK_SIZE,
    //   but will try to always optimize available resources.
    wbCheck(cudaMalloc((void**)&deviceAuxArray, nblocks_scan(numElements) * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceAuxScannedArray, nblocks_scan(numElements) * sizeof(float)));
    wbTime_stop(GPU, "Allocating device memory.");

    wbTime_start(GPU, "Clearing output device memory.");
    //@@ zero out the deviceOutput using cudaMemset() by uncommenting the below line
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output device memory.");

    wbTime_start(GPU, "Copying input host memory to device.");
    //@@ Copy input host memory to device
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input host memory to device.");

    //@@ Initialize the grid and block dimensions here
    // First scan launch
    dim3 DimGrid(nblocks_scan(numElements), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    //@@ You need to launch scan kernel twice: 1) for generating scanned blocks
    //@@ (hint: pass deviceAuxArray to the aux parameter)
    //@@ and 2) for generating scanned aux array that has the scanned block sums.
    //@@ (hint: pass NULL to the aux parameter)
    //@@ Then you should call addScannedBlockSums kernel.
    //invoke CUDA kernel
    scan << < DimGrid, DimBlock >> > (
        deviceInput,
        deviceOutput,
        deviceAuxArray,
        numElements
        );
    //cudaDeviceSynchronize();
    // According to our assumption, the number of blocks for the first scan launch
    //   is less or equal than 2*BLOCK_SIZE, then, for the second launch, we set
    //   a new grid with only one block sized equal to the number of blocks
    //   of the first launch.
    dim3 DimGridAux(1, 1, 1);
    dim3 DimBlockAux(nblocks_scan(numElements) / 2, 1, 1);

    scan << < DimGridAux, DimBlockAux >> > (
        deviceAuxArray,
        deviceAuxScannedArray,
        NULL,
        nblocks_scan(numElements)
        );
    //cudaDeviceSynchronize();

    // For the last launch, we will have one less block compared to the first launch.
    //   The block size must be 2*BLOCK_SIZE because it was the number of elements
    //   processed by the scan.
    dim3 DimGridAuxSum(nblocks_scan(numElements) - 1, 1, 1);
    dim3 DimBlockAuxSum(2 * BLOCK_SIZE, 1, 1);

    addScannedBlockSums << < DimGridAuxSum, DimBlockAuxSum >> > (
        deviceOutput,
        deviceAuxScannedArray,
        numElements
        );
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output device memory to host");
    //@@ Copy results from device to host
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output device memory to host");

    wbTime_start(GPU, "Freeing device memory");
    //@@ Deallocate device memory
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceOutput));
    wbCheck(cudaFree(deviceAuxArray));
    wbCheck(cudaFree(deviceAuxScannedArray));
    wbTime_stop(GPU, "Freeing device memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
