/********************* GPUJIT.c - GPUJIT Execution Engine *********************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is dual licensed under the MIT and the University of Illinois    */
/* Open Source License. See LICENSE.TXT for details.                          */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file implements GPUJIT, a ptx string execution engine for GPU.       */
/*                                                                            */
/******************************************************************************/

#include "GPUJIT.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>

#include "drvapi_error_string.h"

/* Define Polly's GPGPU data types. */
struct PollyGPUContextT {
  CUcontext Cuda;
};

struct PollyGPUModuleT {
  CUmodule Cuda;
};

struct PollyGPUFunctionT {
  CUfunction Cuda;
};

struct PollyGPULinkStateT {
  CUlinkState Cuda;
};

struct PollyGPUDeviceT {
  CUdevice Cuda;
};

struct PollyGPUDevicePtrT {
  CUdeviceptr Cuda;
};

struct PollyGPUEventT {
  cudaEvent_t Cuda;
};

/* Dynamic library handles for the CUDA and CUDA runtime library. */
static void *HandleCuda;
static void *HandleCudaRT;

/* Type-defines of function pointer to CUDA driver APIs. */
typedef CUresult CUDAAPI CuMemAllocFcnTy(CUdeviceptr *, size_t);
static CuMemAllocFcnTy *CuMemAllocFcnPtr;

typedef CUresult CUDAAPI CuFuncSetBlockShapeFcnTy(CUfunction, int, int, int);
static CuFuncSetBlockShapeFcnTy *CuFuncSetBlockShapeFcnPtr;

typedef CUresult CUDAAPI
CuParamSetvFcnTy(CUfunction, int, void *, unsigned int);
static CuParamSetvFcnTy *CuParamSetvFcnPtr;

typedef CUresult CUDAAPI CuParamSetSizeFcnTy(CUfunction, unsigned int);
static CuParamSetSizeFcnTy *CuParamSetSizeFcnPtr;

typedef CUresult CUDAAPI CuLaunchGridFcnTy(CUfunction, int, int);
static CuLaunchGridFcnTy *CuLaunchGridFcnPtr;

typedef CUresult CUDAAPI CuMemcpyDtoHFcnTy(void *, CUdeviceptr, size_t);
static CuMemcpyDtoHFcnTy *CuMemcpyDtoHFcnPtr;

typedef CUresult CUDAAPI CuMemcpyHtoDFcnTy(CUdeviceptr, const void *, size_t);
static CuMemcpyHtoDFcnTy *CuMemcpyHtoDFcnPtr;

typedef CUresult CUDAAPI CuMemFreeFcnTy(CUdeviceptr);
static CuMemFreeFcnTy *CuMemFreeFcnPtr;

typedef CUresult CUDAAPI CuModuleUnloadFcnTy(CUmodule);
static CuModuleUnloadFcnTy *CuModuleUnloadFcnPtr;

typedef CUresult CUDAAPI CuCtxDestroyFcnTy(CUcontext);
static CuCtxDestroyFcnTy *CuCtxDestroyFcnPtr;

typedef CUresult CUDAAPI CuInitFcnTy(unsigned int);
static CuInitFcnTy *CuInitFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetCountFcnTy(int *);
static CuDeviceGetCountFcnTy *CuDeviceGetCountFcnPtr;

typedef CUresult CUDAAPI CuCtxCreateFcnTy(CUcontext *, unsigned int, CUdevice);
static CuCtxCreateFcnTy *CuCtxCreateFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetFcnTy(CUdevice *, int);
static CuDeviceGetFcnTy *CuDeviceGetFcnPtr;

typedef CUresult CUDAAPI
CuModuleLoadDataFcnTy(CUmodule *module, const void *image);
static CuModuleLoadDataFcnTy *CuModuleLoadDataFcnPtr;

typedef CUresult CUDAAPI CuModuleLoadDataExFcnTy(CUmodule *, const void *,
                                                 unsigned int, CUjit_option *,
                                                 void **);
static CuModuleLoadDataExFcnTy *CuModuleLoadDataExFcnPtr;

typedef CUresult CUDAAPI
CuModuleGetFunctionFcnTy(CUfunction *, CUmodule, const char *);
static CuModuleGetFunctionFcnTy *CuModuleGetFunctionFcnPtr;

typedef CUresult CUDAAPI CuDeviceComputeCapabilityFcnTy(int *, int *, CUdevice);
static CuDeviceComputeCapabilityFcnTy *CuDeviceComputeCapabilityFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetNameFcnTy(char *, int, CUdevice);
static CuDeviceGetNameFcnTy *CuDeviceGetNameFcnPtr;

typedef CUresult CUDAAPI
CuLinkAddDataFcnTy(CUlinkState state, CUjitInputType type, void *data,
                   size_t size, const char *name, unsigned int numOptions,
                   CUjit_option *options, void **optionValues);
static CuLinkAddDataFcnTy *CuLinkAddDataFcnPtr;

typedef CUresult CUDAAPI
CuLinkCreateFcnTy(unsigned int numOptions, CUjit_option *options,
                  void **optionValues, CUlinkState *stateOut);
static CuLinkCreateFcnTy *CuLinkCreateFcnPtr;

typedef CUresult CUDAAPI
CuLinkCompleteFcnTy(CUlinkState state, void **cubinOut, size_t *sizeOut);
static CuLinkCompleteFcnTy *CuLinkCompleteFcnPtr;

typedef CUresult CUDAAPI CuLinkDestroyFcnTy(CUlinkState state);
static CuLinkDestroyFcnTy *CuLinkDestroyFcnPtr;

/* Type-defines of function pointer ot CUDA runtime APIs. */
typedef cudaError_t CUDARTAPI CudaEventCreateFcnTy(cudaEvent_t *);
static CudaEventCreateFcnTy *CudaEventCreateFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventRecordFcnTy(cudaEvent_t, cudaStream_t);
static CudaEventRecordFcnTy *CudaEventRecordFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventSynchronizeFcnTy(cudaEvent_t);
static CudaEventSynchronizeFcnTy *CudaEventSynchronizeFcnPtr;

typedef cudaError_t CUDARTAPI
CudaEventElapsedTimeFcnTy(float *, cudaEvent_t, cudaEvent_t);
static CudaEventElapsedTimeFcnTy *CudaEventElapsedTimeFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventDestroyFcnTy(cudaEvent_t);
static CudaEventDestroyFcnTy *CudaEventDestroyFcnPtr;

typedef cudaError_t CUDARTAPI CudaThreadSynchronizeFcnTy(void);
static CudaThreadSynchronizeFcnTy *CudaThreadSynchronizeFcnPtr;

typedef cudaError_t CUDARTAPI CudaDeviceResetFcnTy(void);
CudaDeviceResetFcnTy *CudaDeviceResetFcnPtr;

static void *getAPIHandle(void *Handle, const char *FuncName) {
  char *Err;
  void *FuncPtr;
  dlerror();
  FuncPtr = dlsym(Handle, FuncName);
  if ((Err = dlerror()) != 0) {
    fprintf(stdout, "Load CUDA driver API failed: %s. \n", Err);
    return 0;
  }
  return FuncPtr;
}

static int initialDeviceAPILibraries() {
  HandleCuda = dlopen("libcuda.so", RTLD_LAZY);
  if (!HandleCuda) {
    printf("Cannot open library: %s. \n", dlerror());
    return 0;
  }

  HandleCudaRT = dlopen("libcudart.so", RTLD_LAZY);
  if (!HandleCudaRT) {
    printf("Cannot open library: %s. \n", dlerror());
    return 0;
  }

  return 1;
}

static int initialDeviceAPIs() {
  if (initialDeviceAPILibraries() == 0)
    return 0;

  /* Get function pointer to CUDA Driver APIs.
   *
   * Note that compilers conforming to the ISO C standard are required to
   * generate a warning if a conversion from a void * pointer to a function
   * pointer is attempted as in the following statements. The warning
   * of this kind of cast may not be emitted by clang and new versions of gcc
   * as it is valid on POSIX 2008.
   */
  CuFuncSetBlockShapeFcnPtr = (CuFuncSetBlockShapeFcnTy *)getAPIHandle(
      HandleCuda, "cuFuncSetBlockShape");

  CuParamSetvFcnPtr =
      (CuParamSetvFcnTy *)getAPIHandle(HandleCuda, "cuParamSetv");

  CuParamSetSizeFcnPtr =
      (CuParamSetSizeFcnTy *)getAPIHandle(HandleCuda, "cuParamSetSize");

  CuLaunchGridFcnPtr =
      (CuLaunchGridFcnTy *)getAPIHandle(HandleCuda, "cuLaunchGrid");

  CuMemAllocFcnPtr =
      (CuMemAllocFcnTy *)getAPIHandle(HandleCuda, "cuMemAlloc_v2");

  CuMemFreeFcnPtr = (CuMemFreeFcnTy *)getAPIHandle(HandleCuda, "cuMemFree_v2");

  CuMemcpyDtoHFcnPtr =
      (CuMemcpyDtoHFcnTy *)getAPIHandle(HandleCuda, "cuMemcpyDtoH_v2");

  CuMemcpyHtoDFcnPtr =
      (CuMemcpyHtoDFcnTy *)getAPIHandle(HandleCuda, "cuMemcpyHtoD_v2");

  CuModuleUnloadFcnPtr =
      (CuModuleUnloadFcnTy *)getAPIHandle(HandleCuda, "cuModuleUnload");

  CuCtxDestroyFcnPtr =
      (CuCtxDestroyFcnTy *)getAPIHandle(HandleCuda, "cuCtxDestroy");

  CuInitFcnPtr = (CuInitFcnTy *)getAPIHandle(HandleCuda, "cuInit");

  CuDeviceGetCountFcnPtr =
      (CuDeviceGetCountFcnTy *)getAPIHandle(HandleCuda, "cuDeviceGetCount");

  CuDeviceGetFcnPtr =
      (CuDeviceGetFcnTy *)getAPIHandle(HandleCuda, "cuDeviceGet");

  CuCtxCreateFcnPtr =
      (CuCtxCreateFcnTy *)getAPIHandle(HandleCuda, "cuCtxCreate_v2");

  CuModuleLoadDataFcnPtr =
      (CuModuleLoadDataFcnTy *)getAPIHandle(HandleCuda, "cuModuleLoadData");

  CuModuleLoadDataExFcnPtr =
      (CuModuleLoadDataExFcnTy *)getAPIHandle(HandleCuda, "cuModuleLoadDataEx");

  CuModuleGetFunctionFcnPtr = (CuModuleGetFunctionFcnTy *)getAPIHandle(
      HandleCuda, "cuModuleGetFunction");

  CuDeviceComputeCapabilityFcnPtr =
      (CuDeviceComputeCapabilityFcnTy *)getAPIHandle(
          HandleCuda, "cuDeviceComputeCapability");

  CuDeviceGetNameFcnPtr =
      (CuDeviceGetNameFcnTy *)getAPIHandle(HandleCuda, "cuDeviceGetName");

  CuLinkAddDataFcnPtr =
      (CuLinkAddDataFcnTy *)getAPIHandle(HandleCuda, "cuLinkAddData");

  CuLinkCreateFcnPtr =
      (CuLinkCreateFcnTy *)getAPIHandle(HandleCuda, "cuLinkCreate");

  CuLinkCompleteFcnPtr =
      (CuLinkCompleteFcnTy *)getAPIHandle(HandleCuda, "cuLinkComplete");

  CuLinkDestroyFcnPtr =
      (CuLinkDestroyFcnTy *)getAPIHandle(HandleCuda, "cuLinkDestroy");

  /* Get function pointer to CUDA Runtime APIs. */
  CudaEventCreateFcnPtr =
      (CudaEventCreateFcnTy *)getAPIHandle(HandleCudaRT, "cudaEventCreate");

  CudaEventRecordFcnPtr =
      (CudaEventRecordFcnTy *)getAPIHandle(HandleCudaRT, "cudaEventRecord");

  CudaEventSynchronizeFcnPtr = (CudaEventSynchronizeFcnTy *)getAPIHandle(
      HandleCudaRT, "cudaEventSynchronize");

  CudaEventElapsedTimeFcnPtr = (CudaEventElapsedTimeFcnTy *)getAPIHandle(
      HandleCudaRT, "cudaEventElapsedTime");

  CudaEventDestroyFcnPtr =
      (CudaEventDestroyFcnTy *)getAPIHandle(HandleCudaRT, "cudaEventDestroy");

  CudaThreadSynchronizeFcnPtr = (CudaThreadSynchronizeFcnTy *)getAPIHandle(
      HandleCudaRT, "cudaThreadSynchronize");

  CudaDeviceResetFcnPtr =
      (CudaDeviceResetFcnTy *)getAPIHandle(HandleCudaRT, "cudaDeviceReset");

  return 1;
}

void polly_initDevice(PollyGPUContext **Context, PollyGPUDevice **Device) {
  int Major = 0, Minor = 0, DeviceID = 0;
  char DeviceName[256];
  int DeviceCount = 0;

  /* Get API handles. */
  if (initialDeviceAPIs() == 0) {
    fprintf(stdout, "Getting the \"handle\" for the CUDA driver API failed.\n");
    exit(-1);
  }

  if (CuInitFcnPtr(0) != CUDA_SUCCESS) {
    fprintf(stdout, "Initializing the CUDA driver API failed.\n");
    exit(-1);
  }

  /* Get number of devices that supports CUDA. */
  CuDeviceGetCountFcnPtr(&DeviceCount);
  if (DeviceCount == 0) {
    fprintf(stdout, "There is no device supporting CUDA.\n");
    exit(-1);
  }

  /* We select the 1st device as default. */
  *Device = malloc(sizeof(PollyGPUDevice));
  if (*Device == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU device failed.\n");
    exit(-1);
  }
  CuDeviceGetFcnPtr(&((*Device)->Cuda), DeviceID);

  /* Get compute capabilities and the device name. */
  CuDeviceComputeCapabilityFcnPtr(&Major, &Minor, (*Device)->Cuda);
  CuDeviceGetNameFcnPtr(DeviceName, 256, (*Device)->Cuda);
  fprintf(stderr, "> Running on GPU device %d : %s.\n", DeviceID, DeviceName);

  /* Create context on the device. */
  *Context = malloc(sizeof(PollyGPUContext));
  if (*Context == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU context failed.\n");
    exit(-1);
  }
  CuCtxCreateFcnPtr(&((*Context)->Cuda), 0, (*Device)->Cuda);
}

void polly_getPTXModule(char *PTXBuffer, const char *KernelName,
                        PollyGPUModule **Module, PollyGPUFunction **Kernel) {
  CUresult Res;
  CUlinkState lState;
  CUjit_option options[6];
  void *optionVals[6];
  float walltime;
  char error_log[8192], info_log[8192];
  unsigned int logSize = 8192;
  void *cuOut;
  size_t outSize;

  // Setup linker options
  // Return walltime from JIT compilation
  options[0] = CU_JIT_WALL_TIME;
  optionVals[0] = (void *)&walltime;
  // Pass a buffer for info messages
  options[1] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[1] = (void *)info_log;
  // Pass the size of the info buffer
  options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[2] = (void *)logSize;
  // Pass a buffer for error message
  options[3] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[3] = (void *)error_log;
  // Pass the size of the error buffer
  options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[4] = (void *)logSize;
  // Make the linker verbose
  options[5] = CU_JIT_LOG_VERBOSE;
  optionVals[5] = (void *)1;

  CuLinkCreateFcnPtr(6, options, optionVals, &lState);
  Res = CuLinkAddDataFcnPtr(lState, CU_JIT_INPUT_PTX, (void *)PTXBuffer,
                            strlen(PTXBuffer) + 1, 0, 0, 0, 0);
  if (Res != CUDA_SUCCESS) {
    // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option
    // above.
    fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
    exit(-1);
  }

  // Complete the linker step
  Res = CuLinkCompleteFcnPtr(lState, &cuOut, &outSize);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Complete ptx linker step failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }

  // Linker walltime and info_log were requested in options above.
  printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime,
         info_log);

  // Load resulting cuBin into module
  *Module = malloc(sizeof(PollyGPUModule));
  if (*Module == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU module failed.\n");
    exit(-1);
  }

  Res = CuModuleLoadDataFcnPtr(&((*Module)->Cuda), cuOut);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Loading ptx assembly text failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }

  /* Locate the kernel entry point. */
  *Kernel = malloc(sizeof(PollyGPUFunction));
  if (*Kernel == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU kernel function failed.\n");
    exit(-1);
  }

  Res = CuModuleGetFunctionFcnPtr(&((*Kernel)->Cuda), (*Module)->Cuda,
                                  KernelName);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Loading kernel function failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }

  // Destroy the linker invocation
  CuLinkDestroyFcnPtr(lState);
}

void polly_getPTXKernelEntry(const char *KernelName, PollyGPUModule *Module,
                             PollyGPUFunction **Kernel) {
  CUresult Res;

  *Kernel = malloc(sizeof(PollyGPUFunction));
  if (*Kernel == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU kernel failed.\n");
    exit(-1);
  }

  /* Locate the kernel entry point. */
  Res = CuModuleGetFunctionFcnPtr(&((*Kernel)->Cuda), Module->Cuda, KernelName);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Loading kernel function failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }
}

void polly_startTimerByCudaEvent(PollyGPUEvent **Start, PollyGPUEvent **Stop) {
  *Start = malloc(sizeof(PollyGPUEvent));
  if (*Start == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU start timer failed.\n");
    exit(-1);
  }
  CudaEventCreateFcnPtr(&((*Start)->Cuda));

  *Stop = malloc(sizeof(PollyGPUEvent));
  if (*Stop == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU stop timer failed.\n");
    exit(-1);
  }
  CudaEventCreateFcnPtr(&((*Stop)->Cuda));

  /* Record the start time. */
  CudaEventRecordFcnPtr((*Start)->Cuda, 0);
}

void polly_stopTimerByCudaEvent(PollyGPUEvent *Start, PollyGPUEvent *Stop,
                                float *ElapsedTimes) {
  /* Record the end time. */
  CudaEventRecordFcnPtr(Stop->Cuda, 0);
  CudaEventSynchronizeFcnPtr(Start->Cuda);
  CudaEventSynchronizeFcnPtr(Stop->Cuda);
  CudaEventElapsedTimeFcnPtr(ElapsedTimes, Start->Cuda, Stop->Cuda);
  CudaEventDestroyFcnPtr(Start->Cuda);
  CudaEventDestroyFcnPtr(Stop->Cuda);
  fprintf(stderr, "Processing time: %f (ms).\n", *ElapsedTimes);

  free(Start);
  free(Stop);
}

void polly_initDevDataArray(PollyGPUDevicePtr ***DevDataArray) {
  // We assume total number of arrays and parameters on device is less than 10.
  *DevDataArray = malloc(10 * sizeof(PollyGPUDevicePtr *));
  if (*DevDataArray == 0) {
    fprintf(stdout, "Allocate memory for GPU device data array failed.\n");
    exit(-1);
  }
}

void polly_allocateMemoryForHostAndDevice(void **HostData,
                                          PollyGPUDevicePtr **DevDataArray,
                                          PollyGPUDevicePtr **DevData,
                                          int MemSize, int *NumDevData) {
  assert((*NumDevData) < 10 &&
         "Times of allocate memory on device should be less thean 10.");

  if ((*HostData = (int *)malloc(MemSize)) == 0) {
    fprintf(stdout, "Could not allocate host memory.\n");
    exit(-1);
  }

  *DevData = malloc(sizeof(PollyGPUDevicePtr));
  if (*DevData == 0) {
    fprintf(stdout, "Allocate memory for GPU device memory pointer failed.\n");
    exit(-1);
  }
  CuMemAllocFcnPtr(&((*DevData)->Cuda), MemSize);

  DevDataArray[(*NumDevData)++] = *DevData;
}

void polly_allocateMemoryForDevice(PollyGPUDevicePtr **DevDataArray,
                                   PollyGPUDevicePtr **DevData, int MemSize,
                                   int *NumDevData) {
  CUresult Res;

  assert((*NumDevData) < 10 &&
         "Times of allocate memory on device should be less thean 10.");

  *DevData = malloc(sizeof(PollyGPUDevicePtr));
  if (*DevData == 0) {
    fprintf(stdout, "Allocate memory for GPU device memory pointer failed.\n");
    exit(-1);
  }

  Res = CuMemAllocFcnPtr(&((*DevData)->Cuda), MemSize);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Allocate memory for GPU device memory pointer failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }

  DevDataArray[(*NumDevData)++] = *DevData;
}

void polly_copyFromHostToDevice(PollyGPUDevicePtr *DevData, void *HostData,
                                int MemSize) {
  CUdeviceptr CuDevData = DevData->Cuda;
  CUresult Res = CuMemcpyHtoDFcnPtr(CuDevData, HostData, MemSize);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Copying results from host to device memory failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }
}

void polly_copyFromDeviceToHost(void *HostData, PollyGPUDevicePtr *DevData,
                                int MemSize) {
  CUdeviceptr CuDevData = DevData->Cuda;
  CUresult Res = CuMemcpyDtoHFcnPtr(HostData, CuDevData, MemSize);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Copying results from device to host memory failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }
}

void polly_setKernelParameters(PollyGPUFunction *Kernel,
                               PollyGPUDevicePtr *DevData, int *ParamOffset) {
  void *Ptr;

  Ptr = (void *)DevData->Cuda;
  *ParamOffset = (*ParamOffset + __alignof(Ptr) - 1) & ~(__alignof(Ptr) - 1);
  CuParamSetvFcnPtr(Kernel->Cuda, *ParamOffset, &(DevData->Cuda),
                    sizeof(DevData->Cuda));
  *ParamOffset += sizeof(DevData->Cuda);
  CuParamSetSizeFcnPtr(Kernel->Cuda, *ParamOffset);
}

void polly_setBlockShape(PollyGPUFunction *Kernel, int BlockWidth,
                         int BlockHeight, int BlockDepth) {
  CuFuncSetBlockShapeFcnPtr(Kernel->Cuda, BlockWidth, BlockHeight, BlockDepth);
}

void polly_launchKernel(PollyGPUFunction *Kernel, int GridWidth,
                        int GridHeight) {
  CUresult Res = CuLaunchGridFcnPtr(Kernel->Cuda, GridWidth, GridHeight);
  if (Res != CUDA_SUCCESS) {
    fprintf(stdout, "Launching CUDA kernel failed.\n");
    fprintf(stdout, "The Error Message is %s.\n", getCudaDrvErrorString(Res));
    exit(-1);
  }
  CudaThreadSynchronizeFcnPtr();
  fprintf(stdout, "CUDA kernel launched.\n");
}

void polly_freeDeviceMemory(PollyGPUDevicePtr **DevDataArray, int NumDevData) {
  int i;

  for (i = 0; i < NumDevData; ++i) {
    PollyGPUDevicePtr *DevData = DevDataArray[i];
    if (DevData->Cuda) {
      CuMemFreeFcnPtr(DevData->Cuda);
      DevData->Cuda = 0;
      free(DevData);
      DevData = 0;
    }
  }

  free(DevDataArray);
  DevDataArray = NULL;
}

void polly_cleanupGPGPUResources(PollyGPUModule *Module,
                                 PollyGPUContext *Context,
                                 PollyGPUFunction *Kernel,
                                 PollyGPUDevice *Device) {
  if (Module->Cuda) {
    CuModuleUnloadFcnPtr(Module->Cuda);
    Module->Cuda = 0;
    free(Module);
    Module = 0;
  }

  if (Context->Cuda) {
    CuCtxDestroyFcnPtr(Context->Cuda);
    Context->Cuda = 0;
    free(Context);
    Context = 0;
  }

  if (Kernel) {
    free(Kernel);
    Kernel = 0;
  }

  if (Device) {
    free(Device);
    Device = 0;
  }

  CudaDeviceResetFcnPtr();

  dlclose(HandleCuda);
  dlclose(HandleCudaRT);
}
