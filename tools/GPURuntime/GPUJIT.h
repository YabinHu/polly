/******************************************************************************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is dual licensed under the MIT and the University of Illinois    */
/* Open Source License. See LICENSE.TXT for details.                          */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file defines GPUJIT.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef GPUJIT_H_
#define GPUJIT_H_

/*
 * The following demostrates how we can use the GPURuntime library to
 * execute a GPU kernel.
 *
 * char KernelString[] = "\n\
 *   .version 1.4\n\
 *   .target sm_10, map_f64_to_f32\n\
 *   .entry _Z8myKernelPi (\n\
 *   .param .u64 __cudaparm__Z8myKernelPi_data)\n\
 *   {\n\
 *     .reg .u16 %rh<4>;\n\
 *     .reg .u32 %r<5>;\n\
 *     .reg .u64 %rd<6>;\n\
 *     cvt.u32.u16     %r1, %tid.x;\n\
 *     mov.u16         %rh1, %ctaid.x;\n\
 *     mov.u16         %rh2, %ntid.x;\n\
 *     mul.wide.u16    %r2, %rh1, %rh2;\n\
 *     add.u32         %r3, %r1, %r2;\n\
 *     ld.param.u64    %rd1, [__cudaparm__Z8myKernelPi_data];\n\
 *     cvt.s64.s32     %rd2, %r3;\n\
 *     mul.wide.s32    %rd3, %r3, 4;\n\
 *     add.u64         %rd4, %rd1, %rd3;\n\
 *     st.global.s32   [%rd4+0], %r3;\n\
 *     exit;\n\
 *   }\n\
 * ";
 *
 * const char *Entry = "_Z8myKernelPi";
 *
 * int main() {
 *   PollyGPUContext *Context;
 *   PollyGPUModule *Module;
 *   PollyGPUFunction *Kernel;
 *   PollyGPUDevice *Device;
 *   PollyGPUDevicePtr **DevDataArray;
 *   PollyGPUDevicePtr *DevData;
 *   int *HostData;
 *   PollyGPUEvent *Start;
 *   PollyGPUEvent *Stop;
 *   float ElapsedTime = 0.0;
 *   int MemSize;
 *   int BlockWidth = 16;
 *   int BlockHeight = 16;
 *   int BlockDepth = 1;
 *   int GridWidth = 8;
 *   int GridHeight = 8;
 *   int NumDevData = 0;
 *   int ParamOffset = 0;
 *
 *   MemSize = 256*64*sizeof(int);
 *   polly_initDevice(&Context, &Device);
 *   polly_getPTXModule(KernelString, &Module);
 *   polly_getPTXKernelEntry(Entry, Module, &Kernel);
 *   HostData = (int *)malloc(MemSize);
 *   polly_initDevDataArray(&DevDataArray);
 *   polly_allocateMemoryForDevice(DevDataArray, &DevData, MemSize,
 *                                 &NumDevData);
 *   polly_setKernelParameters(Kernel, DevData, &ParamOffset);
 *   polly_setBlockShape(Kernel, BlockWidth, BlockHeight, BlockDepth);
 *   polly_startTimerByCudaEvent(&Start, &Stop);
 *   polly_launchKernel(Kernel, GridWidth, GridHeight);
 *   polly_copyFromDeviceToHost(HostData, DevData, MemSize);
 *   polly_stopTimerByCudaEvent(Start, Stop, &ElapsedTime);
 *   polly_freeDeviceMemory(DevDataArray, NumDevData);
 *   polly_cleanupGPGPUResources(DevData, Module, Context, Kernel, Device);
 *   free(HostData);
 * }
 *
 */

typedef struct PollyGPUContextT PollyGPUContext;
typedef struct PollyGPUModuleT PollyGPUModule;
typedef struct PollyGPUFunctionT PollyGPUFunction;
typedef struct PollyGPUDeviceT PollyGPUDevice;
typedef struct PollyGPUDevicePtrT PollyGPUDevicePtr;
typedef struct PollyGPUEventT PollyGPUEvent;

void polly_initDevice(PollyGPUContext **Context, PollyGPUDevice **Device);
void polly_getPTXModule(char *PTXBuffer, const char *KernelName,
                        PollyGPUModule **Module, PollyGPUFunction **Kernel);
void polly_getPTXKernelEntry(const char *KernelName, PollyGPUModule *Module,
                             PollyGPUFunction **Kernel);
void polly_startTimerByCudaEvent(PollyGPUEvent **Start, PollyGPUEvent **Stop);
void polly_stopTimerByCudaEvent(PollyGPUEvent *Start, PollyGPUEvent *Stop,
                                float *ElapsedTimes);
void polly_copyFromHostToDevice(PollyGPUDevicePtr *DevData, void *HostData,
                                int MemSize);
void polly_copyFromDeviceToHost(void *HostData, PollyGPUDevicePtr *DevData,
                                int MemSize);
void polly_initDevDataArray(PollyGPUDevicePtr ***DevDataArray);
void polly_allocateMemoryForHostAndDevice(void **HostData,
                                          PollyGPUDevicePtr **DevDataArray,
                                          PollyGPUDevicePtr **DevData,
                                          int MemSize, int *NumDevData);
void polly_allocateMemoryForDevice(PollyGPUDevicePtr **DevDataArray,
                                   PollyGPUDevicePtr **DevData, int MemSize,
                                   int *NumDevData);
void polly_setKernelParameters(PollyGPUFunction *Kernel,
                               PollyGPUDevicePtr *DevData, int *ParamOffset);
void polly_setBlockShape(PollyGPUFunction *Kernel, int BlockWidth,
                         int BlockHeight, int BlockDepth);
void polly_launchKernel(PollyGPUFunction *Kernel, int GridWidth,
                        int GridHeight);
void polly_freeDeviceMemory(PollyGPUDevicePtr **DevDataArray, int NumDevData);
void polly_cleanupGPGPUResources(PollyGPUModule *Module,
                                 PollyGPUContext *Context,
                                 PollyGPUFunction *Kernel,
                                 PollyGPUDevice *Device);
#endif /* GPUJIT_H_ */
