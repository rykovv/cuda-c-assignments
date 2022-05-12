#include <wb.h>
#include <CL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WORK_GROUP_SIZE 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cl_int err = stmt;                                                    \
    if (err != CL_SUCCESS) {                                              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got OPENCL error ...  ", get_error_string(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Write the OpenCL kernel
const char *kernelSource =	"__kernel void vadd (__global const float *a, __global const float *b, __global float *result) {\n"
								                "int id = get_global_id(0);\n"
								                "result[id] = a[id] + b[id];\n"
                            "};";

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int inputLengthBytes;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;
  deviceInput1 = NULL;
  deviceInput2 = NULL;
  deviceOutput = NULL;

  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id;    // device ID
  cl_context context;        // context
  cl_command_queue queue;    // command queue
  cl_program program;        // program
  cl_kernel kernel;          // kernel

  context = NULL;
  queue = NULL;
  program = NULL;
  kernel = NULL;

  cl_int clerr = CL_SUCCESS;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  inputLengthBytes = inputLength * sizeof(float);
  hostOutput       = (float *)malloc(inputLengthBytes);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The input size is ", inputLengthBytes, " bytes");

  //@@ Initialize the workgroup dimensions
  int n_work_groups = (inputLength - 1) / WORK_GROUP_SIZE + 1;
  //@@ Bind to platform
  wbCheck(clGetPlatformIDs(1, &cpPlatform, NULL));
  //@@ Get ID for the device
  wbCheck(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));
  //@@ Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &clerr);
  wbCheck(clerr);
  //@@ Create a command queue
  queue = clCreateCommandQueue(context, device_id, 0, &clerr);
  wbCheck(clerr);
  //@@ Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &clerr);
  wbCheck(clerr);
  //@@ Build the program executable
  wbCheck(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
  //@@ Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "vadd", &clerr);
  wbCheck(clerr);
  //@@ Create the input and output arrays in device memory for our calculation
  //@@ Write our data set into the input array in device memory
  deviceInput1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput1, NULL);
  deviceInput2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput2, NULL);
  deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputLengthBytes, NULL, NULL);

  //@@ Set the arguments to our compute kernel
  wbCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&deviceInput1));
  wbCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&deviceInput2));
  wbCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&deviceOutput));
  //@@ Execute the kernel over the entire range of the data set
  cl_event event = NULL;
  wbCheck(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, n_work_groups, WORK_GROUP_SIZE, 0, NULL, &event));
  //@@ Wait for the command queue to get serviced before reading back results
  wbCheck(clWaitForEvents(1, &event));
  //@@ Read the results from the device
  wbCheck(clEnqueueReadBuffer(queue, deviceOutput, CL_TRUE, 0, inputLengthBytes, hostOutput, 0, NULL, NULL));

  wbSolution(args, hostOutput, inputLength);

  // release OpenCL resources
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
