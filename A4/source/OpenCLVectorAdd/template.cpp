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

const char* get_error_string(cl_int err);

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
  size_t global_group_size = inputLength;
  size_t work_group_size = WORK_GROUP_SIZE/2;
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
  wbCheck(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_group_size, &work_group_size, 0, NULL, &event));
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

const char* get_error_string(cl_int err) {
    switch (err) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";

    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    default: return "Unknown OpenCL error";
    }
}
