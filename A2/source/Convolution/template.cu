#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define MASK_WIDTH		5
#define O_TILE_WIDTH	16
#define BLOCK_WIDTH		(O_TILE_WIDTH + MASK_WIDTH / 2)
#define IN_CHANNELS		3
#define IN_CHANNELS_R	0
#define IN_CHANNELS_G	1
#define IN_CHANNELS_B	2

#define clamp(x) (min(max((x), 0.0), 1.0))

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values
__global__ void convolution_2D_kernel_tiled (
	float* in,
	const float* __restrict__ m,
	float* out,
	int height,
	int width)
{
	// if the device does not have enough shared memory, a phased algorithm
	//   can be elaborated where in each phase one channel is convolved
	__shared__ float ds_in[BLOCK_WIDTH][BLOCK_WIDTH][IN_CHANNELS];

	int tx = threadIdx.x, ty = threadIdx.y;
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;

	int row_i = row_o - MASK_WIDTH / 2;
	int col_i = col_o - MASK_WIDTH / 2;

	float tmpR = 0, tmpG = 0, tmpB = 0;

	if ((row_i >= 0 && row_i < height) &&
		(col_i >= 0 && col_i < width))
	{
		ds_in[ty][tx][IN_CHANNELS_R] = in[(row_i * width + col_i) * IN_CHANNELS + IN_CHANNELS_R];
		ds_in[ty][tx][IN_CHANNELS_G] = in[(row_i * width + col_i) * IN_CHANNELS + IN_CHANNELS_G];
		ds_in[ty][tx][IN_CHANNELS_B] = in[(row_i * width + col_i) * IN_CHANNELS + IN_CHANNELS_B];
	} else {
		ds_in[ty][tx][IN_CHANNELS_R] = 0;
		ds_in[ty][tx][IN_CHANNELS_G] = 0;
		ds_in[ty][tx][IN_CHANNELS_B] = 0;
	}
	__syncthreads();

	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
		for (int i = 0; i < MASK_WIDTH; i++) {
			for (int j = 0; j < MASK_WIDTH; j++) {
				tmpR += m[i * MASK_WIDTH + j] * ds_in[ty + i][tx + j][IN_CHANNELS_R];
				tmpG += m[i * MASK_WIDTH + j] * ds_in[ty + i][tx + j][IN_CHANNELS_G];
				tmpB += m[i * MASK_WIDTH + j] * ds_in[ty + i][tx + j][IN_CHANNELS_B];
			}
		}
	}
	__syncthreads();

	if (row_o < height && col_o < width) {
		out[(row_o * width + col_o) * IN_CHANNELS + IN_CHANNELS_R] = clamp(tmpR);
		out[(row_o * width + col_o) * IN_CHANNELS + IN_CHANNELS_G] = clamp(tmpG);
		out[(row_o * width + col_o) * IN_CHANNELS + IN_CHANNELS_B] = clamp(tmpB);
	}
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //allocate device memory
  wbCheck(cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceMaskData, maskRows * maskColumns * sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  //copy host memory to device
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  //initialize thread block and kernel grid dimensions
  dim3 DimGrid((wbImage_getWidth(inputImage) - 1) / O_TILE_WIDTH + 1, (wbImage_getHeight(inputImage) - 1) / O_TILE_WIDTH + 1, 1);
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  //invoke CUDA kernel
  convolution_2D_kernel_tiled <<< DimGrid, DimBlock >>> (
	  deviceInputImageData,
	  deviceMaskData,
	  deviceOutputImageData,
	  imageHeight,
	  imageWidth
  );
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  //copy results from device to host
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ INSERT CODE HERE
  //deallocate device memory
  wbCheck(cudaFree(deviceInputImageData));
  wbCheck(cudaFree(deviceMaskData));
  wbCheck(cudaFree(deviceOutputImageData));

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
