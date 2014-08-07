#ifndef POLLY_CODEGEN_ISL_PTX_GENERATOR_H
#define POLLY_CODEGEN_ISL_PTX_GENERATOR_H

#include "polly/Config/config.h"

#ifdef GPU_CODEGEN
#include "polly/CodeGen/IRBuilder.h"
#include "llvm/ADT/SetVector.h"

#include "isl/ast.h"
#include "isl/map.h"
#include "isl/set.h"
//#include <isl/id_to_ast_expr.h>

#include <map>

namespace llvm {
class Value;
class Pass;
class BasicBlock;
}

struct gpu_array_info;
struct gpu_array_ref_group;
struct gpu_prog;
struct isl_ctx;
struct isl_ast_node;
struct isl_ast_expr;
struct isl_ast_build;
struct ppcg_kernel;
struct ppcg_options;
struct ppcg_scop;

namespace polly {
using namespace llvm;

class IslExprBuilder;
class Scop;
class ScopStmt;

class IslPTXGenerator {
public:
  typedef DenseMap<const Value *, Value *> ValueToValueMapTy;

  IslPTXGenerator(PollyIRBuilder &Builder, IslExprBuilder &ExprBuilder, Pass *P,
                  const std::string &Triple, struct ppcg_options *&Opt);

  ~IslPTXGenerator();

  /// @brief Get the guard of generated AST node for host code.
  __isl_give isl_ast_node *getHostGuard() {
    return isl_ast_node_copy(Guard);
  }

  /// @brief Get the generated isl AST for GPGPU.
  __isl_give isl_ast_node *getOutputAST() { return isl_ast_node_copy(Tree); }

  /// @brief Get the options for GPGPU code generation.
  struct ppcg_options *getOptions() { return Options; }

  /// @brief Create a GPGPU parallel loop.
  ///
  /// @param CurKernel    This is the current kernel.
  /// @param KernelBody   A pointer to an iterator that is set to point to the
  ///                     body of the created loop. It should be used to insert
  ///                     instructions that form the actual loop body.
  void startGeneration(struct ppcg_kernel *CurKernel,
                       BasicBlock::iterator *KernelBody);

  /// @brief Execute the post-operations to build a GPGPU parallel loop.
  ///
  /// @param Subfunction The kernel function in LLVM-IR.
  void finishGeneration(Function *SubFunction);

  /// @brief Set the size of the output array.
  ///
  /// This size is used to allocate memory on the device and the host.
  ///
  /// @param Bytes        Output array size in bytes.
  void setOutputBytes(unsigned Bytes) { OutputBytes = Bytes; }

  /// @brief Set the parameters for launching PTX kernel.
  ///
  void setLaunchingParameters(Value *GridSizeX, Value *GridSizeY);

  /// @brief Add synchronization statement to kernel.
  void addKernelSynchronization();

  /// @brief Get the Value of GPU block id or thread id by Name.
  Value *getValueOfGPUID(const char *Name);

private:
  PollyIRBuilder &Builder;
  IslExprBuilder &ExprBuilder;
  Pass *P;

  /// @brief The target triple of the device.
  const std::string &GPUTriple;

  /// @brief Options for GPGPU code generation.
  struct ppcg_options *Options;

  /// @brief Parameters used for launching PTX kernel.
  Value *GridDimX, *GridDimY;

  /// @brief Size of the output array in bytes.
  unsigned OutputBytes;

  /// @brief Internal representation of a Scop for GPGPU code generation.
  struct ppcg_scop *Scop;

  /// @brief The generated GPU kernel.
  struct ppcg_kernel *Kernel;

  /// @brief The if-like gurad of the host code.
  isl_ast_node *Guard;

  /// @brief The generated AST.
  isl_ast_node *Tree;

  /// @brief Information about the current GPU program.
  struct gpu_prog *Prog;

  /// @brief All the array base addresses in this Scop.
  SetVector<Value *> BaseAddresses;

  /// @brief Build the internal scop.
  void buildScop();

  /// @brief Cleanup resources in scop.
  void freeScop();

  /// @brief Build the generated gpu kernel.
  void buildGPUKernel();

  Module *getModule();

  polly::Scop *getPollyScop();

  isl_ctx *getIslCtx();

  /// @brief Polly's GPU data types.
  StructType *ContextTy, *ModuleTy, *KernelTy, *DeviceTy, *DevDataTy, *EventTy;

  /// @brief GPU execution block ids and thread ids.
  Value *BIDx, *BIDy, *TIDx, *TIDy, *TIDz;

  void initializeGPUDataTypes();
  IntegerType *getInt64Type();           // i64
  PointerType *getI8PtrType();           // char *
  PointerType *getPtrI8PtrType();        // char **
  PointerType *getFloatPtrType();        // float *
  PointerType *getGPUContextPtrType();   // %struct.PollyGPUContextT *
  PointerType *getGPUModulePtrType();    // %struct.PollyGPUModuleT *
  PointerType *getGPUDevicePtrType();    // %struct.PollyGPUDeviceT *
  PointerType *getPtrGPUDevicePtrType(); // %struct.PollyGPUDevicePtrT *
  PointerType *getGPUFunctionPtrType();  // %struct.PollyGPUFunctionT *
  PointerType *getGPUEventPtrType();     // %struct.PollyGPUEventT *

  void initializeBaseAddresses();

  /// @brief Create the kernel string containing LLVM IR.
  ///
  /// @param SubFunction  A pointer to the device code function.
  /// @return             A global string variable containing the LLVM IR codes
  //                      of the SubFunction.
  Value *createPTXKernelFunction(Function *SubFunction);

  /// @brief Get the entry name of the device kernel function.
  ///
  /// @param SubFunction  A pointer to the device code function.
  /// @return             A global string variable containing the entry name of
  ///                     the SubFunction.
  Value *getPTXKernelEntryName(Function *SubFunction);

  void createCallInitDevice(Value *Context, Value *Device);
  void createCallGetPTXModule(Value *Buffer, Value *Module);
  void createCallGetPTXKernelEntry(Value *Entry, Value *Module, Value *Kernel);
  void createCallAllocateMemoryForDevice(Value *DeviceData, Value *Size);
  void createCallCopyFromHostToDevice(Value *DeviceData, Value *HostData,
                                      Value *Size);
  void createCallCopyFromDeviceToHost(Value *HostData, Value *DeviceData,
                                      Value *Size);
  void createCallSetKernelParameters(Value *Kernel, Value *BlockWidth,
                                     Value *BlockHeight, Value *DeviceData,
                                     Value *ParamOffset);
  void createCallLaunchKernel(Value *Kernel, Value *GridWidth,
                              Value *GridHeight);
  void createCallStartTimerByCudaEvent(Value *StartEvent, Value *StopEvent);
  void createCallStopTimerByCudaEvent(Value *StartEvent, Value *StopEvent,
                                      Value *Timer);
  void createCallCleanupGPGPUResources(Value *DeviceData, Value *Module,
                                       Value *Context, Value *Kernel,
                                       Value *Device);
  void createCallBarrierIntrinsic();

  /// @brief Create the CUDA subfunction.
  ///
  /// @param SubFunction  The newly created SubFunction is returned here.
  void createSubfunction(Function **Subfunction);

  /// @brief Create the definition of the CUDA subfunction.
  /// @param NumMemAccs   The number of memory accesses which will be copied
  //                      from host to device.
  /// @param NumArgs      The number of parameters of this scop.
  Function *createSubfunctionDefinition(int NumMemAccs, int NumArgs);

  /// @brief Get the Value of CUDA block X-dimension.
  Value *getCUDABlockDimX();

  /// @brief Get the Value of CUDA block Y-dimension.
  Value *getCUDABlockDimY();

  /// @brief Get the Value of CUDA block Z-dimension.
  Value *getCUDABlockDimZ();

  /// @brief Get the Value of CUDA grid X-dimension.
  Value *getCUDAGridDimX();

  /// @brief Get the Value of CUDA grid Y-dimension.
  Value *getCUDAGridDimY();

  /// @brief Get the Value of the bytes of the output array.
  Value *getOutputArraySizeInBytes();

  Value *getBaseAddressByName(std::string Name);
  Value *getArraySize(struct gpu_array_info *Array, isl_set *Context);

  /// @brief Erase the ptx-related subfunctions and declarations.
  ///
  /// @param SubFunction  A pointer to the device code function.
  void eraseUnusedFunctions(Function *SubFunction);
};
} // end namespace polly

#endif /* GPU_CODEGEN */
#endif /* POLLY_CODEGEN_ISL_PTX_GENERATOR_H */
