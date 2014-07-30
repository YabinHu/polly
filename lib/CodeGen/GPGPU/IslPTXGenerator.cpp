//===------ IslPTXGenerator.cpp -  IR helper to create loops --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create GPU parallel codes as LLVM-IR.
//
//===----------------------------------------------------------------------===//

/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

//#include <assert.h>
//#include <stdlib.h>
//#include <string.h>

#include "polly/CodeGen/IslPTXGenerator.h"

#ifdef GPU_CODEGEN
#include "polly/Dependences.h"
#include "polly/ScopInfo.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/IslExprBuilder.h"

#include "isl/polynomial.h"
#include "isl/union_set.h"
#include "isl/aff.h"
#include "isl/ast.h"
#include "isl/constraint.h"
#include "isl/ilp.h"
#include "isl/flow.h"
#include "isl/band.h"
#include "isl/schedule.h"
#include "isl/options.h"
#include "isl/ast_build.h"

#include "llvm/PassManager.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <set>

#include "gpu.h"
#include "pet.h"
#include "ppcg.h"
#include "ppcg_options.h"
#include "schedule.h"

using namespace polly;

IslPTXGenerator::IslPTXGenerator(PollyIRBuilder &Builder,
                                 IslExprBuilder &ExprBuilder, Pass *P,
                                 const std::string &Triple,
                                 struct ppcg_options *&Opt)
    : Builder(Builder), ExprBuilder(ExprBuilder), P(P), GPUTriple(Triple),
      Options(Opt), Kernel(nullptr), Guard(nullptr), Tree(nullptr),
      Prog(nullptr) {

  buildScop();
  buildGPUKernel();
  initializeGPUDataTypes();
}

IslPTXGenerator::~IslPTXGenerator() {
  isl_ast_node_free(Guard);
  isl_ast_node_free(Tree);
  gpu_prog_free(Prog);
  freeScop();
}

Module *IslPTXGenerator::getModule() {
  return Builder.GetInsertBlock()->getParent()->getParent();
}

polly::Scop *IslPTXGenerator::getPollyScop() {
  ScopInfo &SI = P->getAnalysis<ScopInfo>();
  return SI.getScop();
}

isl_ctx *IslPTXGenerator::getIslCtx() { return getPollyScop()->getIslCtx(); }

Function *IslPTXGenerator::createSubfunctionDefinition(int NumMemAccs,
                                                       int NumVars) {
  Module *M = getModule();
  Function *F = Builder.GetInsertBlock()->getParent();
  std::vector<Type *> Arguments;

  for (int i = 0; i < NumMemAccs + NumVars; i++)
    Arguments.push_back(Builder.getInt8PtrTy());

  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *FN = Function::Create(FT, Function::InternalLinkage,
                                  F->getName() + "_ptx_subfn", M);
  FN->setCallingConv(CallingConv::PTX_Kernel);

  // Do not run any optimization pass on the new function.
  P->getAnalysis<polly::ScopDetection>().markFunctionAsInvalid(FN);

  int j = 0;
  for (Function::arg_iterator AI = FN->arg_begin(); AI != FN->arg_end(); ++AI) {
    if (j < NumMemAccs) {
      AI->setName("ptx.Array");
      j++;
      continue;
    }

    AI->setName("ptx.Var");
  }

  return FN;
}

void IslPTXGenerator::buildScop() {
  polly::Scop *S = getPollyScop();
  Scop = ppcg_scop_from_pet_scop(S, Options);
  assert(Scop && "Build ppcg scop failed.");
}

static void freeScopArray(struct pet_array *array) {
  if (!array)
    return;

  if (array->context)
    isl_set_free(array->context);
  if (array->extent)
    isl_set_free(array->extent);
  if (array->value_bounds)
    isl_set_free(array->value_bounds);

  free(array);
  array = nullptr;
}

void IslPTXGenerator::freeScop() {
  if (!Scop)
    return;

  int N = Scop->n_array;
  for (int i = 0; i < N; i++)
    freeScopArray(Scop->arrays[i]);
  free(Scop->arrays);
  Scop->arrays = nullptr;

  ppcg_scop_free(Scop);
  Scop = nullptr;
}

void IslPTXGenerator::buildGPUKernel() {
  isl_ctx *Ctx = getIslCtx();
  struct gen_ext ext;
  ext.guard = nullptr;
  ext.tree = nullptr;
  ext.prog = nullptr;
  if (generate_gpu(Ctx, Scop, Options, &ext) < 0) {
    errs() << "GPGPU code geneation failed.\n";
    return;
  }
  Guard = ext.guard;
  Tree = ext.tree;
  Prog = ext.prog;
}

void IslPTXGenerator::createSubfunction(ValueToValueMapTy &VMap,
                                        Function **Subfunction) {

  int NumMemAccs, NumVars;
  assert(Kernel && "Kernel should have been set correctly.");
  NumMemAccs = Kernel->n_array;
  NumVars = Kernel->n_var;

  Function *F = createSubfunctionDefinition(NumMemAccs, NumVars);

  Module *M = getModule();
  LLVMContext &Context = F->getContext();
  IntegerType *Ty = Builder.getInt64Ty();

  // Store the previous basic block.
  BasicBlock *PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "ptx.setup", F);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "ptx.exit", F);
  BasicBlock *BodyBB = BasicBlock::Create(Context, "ptx.kernel_body", F);

  DominatorTree &DT = P->getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(BodyBB, HeaderBB);
  Builder.SetInsertPoint(HeaderBB);

  // Create Value* map of host array to its device counterpart.
  // createValueMap(VMap, Kernel->array);

  // Create Value* map of host scalar variable to its device counterpart.
  // createValueMap(VMap, Kernel->var);

  // Add blockID, threadID, grid size, block size, etc.
  // FIXME: These intrinsics should be inserted on-demand. However, we insert
  // them all currently for simplicity.
  Function *GetNctaidX =
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_nctaid_x);
  Function *GetNctaidY =
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_nctaid_y);
  Function *GetCtaidX =
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_x);
  Function *GetCtaidY =
      Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ctaid_y);
  Function *GetNtidX = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ntid_x);
  Function *GetNtidY = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ntid_y);
  Function *GetNtidZ = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_ntid_z);
  Function *GetTidX = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_x);
  Function *GetTidY = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_y);
  Function *GetTidZ = Intrinsic::getDeclaration(M, Intrinsic::ptx_read_tid_z);

  Value *GridWidth = Builder.CreateCall(GetNctaidX);
  GridWidth = Builder.CreateIntCast(GridWidth, Ty, false);
  Value *GridHeight = Builder.CreateCall(GetNctaidY);
  GridHeight = Builder.CreateIntCast(GridHeight, Ty, false);
  Value *BlockWidth = Builder.CreateCall(GetNtidX);
  BlockWidth = Builder.CreateIntCast(BlockWidth, Ty, false);
  Value *BlockHeight = Builder.CreateCall(GetNtidY);
  BlockHeight = Builder.CreateIntCast(BlockHeight, Ty, false);
  Value *BlockDepth = Builder.CreateCall(GetNtidZ);
  BlockDepth = Builder.CreateIntCast(BlockDepth, Ty, false);

  int NumGrid = isl_multi_pw_aff_dim(Kernel->grid_size, isl_dim_set);
  int NumBlock = Kernel->n_block;

  switch (NumGrid) {
  case 1: {
    BIDx = Builder.CreateCall(GetCtaidX);
    BIDx = Builder.CreateIntCast(BIDx, Ty, false);
    break;
  }
  case 2: {
    BIDx = Builder.CreateCall(GetCtaidX);
    BIDx = Builder.CreateIntCast(BIDx, Ty, false);
    BIDy = Builder.CreateCall(GetCtaidY);
    BIDy = Builder.CreateIntCast(BIDy, Ty, false);
    break;
  }
  default:
    errs() << "Set gird id error\n";
    break;
  }

  switch (NumBlock) {
  case 1: {
    TIDx = Builder.CreateCall(GetTidX);
    TIDx = Builder.CreateIntCast(TIDx, Ty, false);
    break;
  }
  case 2: {
    TIDx = Builder.CreateCall(GetTidX);
    TIDx = Builder.CreateIntCast(TIDx, Ty, false);
    TIDy = Builder.CreateCall(GetTidY);
    TIDy = Builder.CreateIntCast(TIDy, Ty, false);
    break;
  }
  case 3: {
    TIDx = Builder.CreateCall(GetTidX);
    TIDx = Builder.CreateIntCast(TIDx, Ty, false);
    TIDy = Builder.CreateCall(GetTidY);
    TIDy = Builder.CreateIntCast(TIDy, Ty, false);
    TIDz = Builder.CreateCall(GetTidZ);
    TIDz = Builder.CreateIntCast(TIDz, Ty, false);
    break;
  }
  default:
    errs() << "Set thread id error.\n";
    break;
  }

  Builder.CreateBr(BodyBB);
  Builder.SetInsertPoint(BodyBB);

  // Build Value* map for original loop index to its counterpart calculated by
  // GPU blockID and threadID.

  // Allocate array space in shared memory

  // Build Value* map for array element in global memory to its counterpart in
  // shared memory.

  // Create ret block.
  // Add the termination of the ptx-device subfunction.
  Builder.CreateBr(ExitBB);
  Builder.SetInsertPoint(--Builder.GetInsertPoint());
  BasicBlock::iterator KernelBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  // Reset insert point to continuation of the kernel body.
  Builder.SetInsertPoint(KernelBody);
  *Subfunction = F;
}

Value *IslPTXGenerator::getValueOfGPUID(const char *Name) {
  if (!strcmp(Name, "b0"))
    return cast<Value>(BIDx);

  if (!strcmp(Name, "b1"))
    return cast<Value>(BIDy);

  if (!strcmp(Name, "t0"))
    return cast<Value>(TIDx);

  if (!strcmp(Name, "t1"))
    return cast<Value>(TIDy);

  if (!strcmp(Name, "t2"))
    return cast<Value>(TIDz);

  return nullptr;
}

void IslPTXGenerator::startGeneration(struct ppcg_kernel *CurKernel,
                                      ValueToValueMapTy &VMap,
                                      BasicBlock::iterator *KernelBody) {
  Function *SubFunction;
  BasicBlock::iterator PrevInsertPoint = Builder.GetInsertPoint();
  Kernel = CurKernel;
  createSubfunction(VMap, &SubFunction);
  *KernelBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(PrevInsertPoint);
}

IntegerType *IslPTXGenerator::getInt64Type() { return Builder.getInt64Ty(); }

PointerType *IslPTXGenerator::getI8PtrType() {
  return PointerType::getUnqual(Builder.getInt8Ty());
}

PointerType *IslPTXGenerator::getPtrI8PtrType() {
  return PointerType::getUnqual(getI8PtrType());
}

PointerType *IslPTXGenerator::getFloatPtrType() {
  return llvm::Type::getFloatPtrTy(getModule()->getContext());
}

PointerType *IslPTXGenerator::getGPUContextPtrType() {
  return PointerType::getUnqual(ContextTy);
}

PointerType *IslPTXGenerator::getGPUModulePtrType() {
  return PointerType::getUnqual(ModuleTy);
}

PointerType *IslPTXGenerator::getGPUDevicePtrType() {
  return PointerType::getUnqual(DeviceTy);
}

PointerType *IslPTXGenerator::getPtrGPUDevicePtrType() {
  return PointerType::getUnqual(DevDataTy);
}

PointerType *IslPTXGenerator::getGPUFunctionPtrType() {
  return PointerType::getUnqual(KernelTy);
}

PointerType *IslPTXGenerator::getGPUEventPtrType() {
  return PointerType::getUnqual(EventTy);
}

void IslPTXGenerator::initializeGPUDataTypes() {
  LLVMContext &Context = getModule()->getContext();

  StructType *TempTy = getModule()->getTypeByName("struct.PollyGPUContextT");
  if (!TempTy)
    ContextTy = StructType::create(Context, "struct.PollyGPUContextT");
  else if (ContextTy != TempTy)
    ContextTy = TempTy;

  TempTy = getModule()->getTypeByName("struct.PollyGPUModuleT");
  if (!TempTy)
    ModuleTy = StructType::create(Context, "struct.PollyGPUModuleT");
  else if (ModuleTy != TempTy)
    ModuleTy = TempTy;

  TempTy = getModule()->getTypeByName("struct.PollyGPUFunctionT");
  if (!TempTy)
    KernelTy = StructType::create(Context, "struct.PollyGPUFunctionT");
  else if (KernelTy != TempTy)
    KernelTy = TempTy;

  TempTy = getModule()->getTypeByName("struct.PollyGPUDeviceT");
  if (!TempTy)
    DeviceTy = StructType::create(Context, "struct.PollyGPUDeviceT");
  else if (DeviceTy != TempTy)
    DeviceTy = TempTy;

  TempTy = getModule()->getTypeByName("struct.PollyGPUDevicePtrT");
  if (!TempTy)
    DevDataTy = StructType::create(Context, "struct.PollyGPUDevicePtrT");
  else if (DevDataTy != TempTy)
    DevDataTy = TempTy;

  TempTy = getModule()->getTypeByName("struct.PollyGPUEventT");
  if (!TempTy)
    EventTy = StructType::create(Context, "struct.PollyGPUEventT");
  else if (EventTy != TempTy)
    EventTy = TempTy;
}

void IslPTXGenerator::createCallInitDevice(Value *Context, Value *Device) {
  const char *Name = "polly_initDevice";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(PointerType::getUnqual(getGPUContextPtrType()));
    Args.push_back(PointerType::getUnqual(getGPUDevicePtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, Context, Device);
}

void IslPTXGenerator::createCallGetPTXModule(Value *Buffer, Value *Module) {
  const char *Name = "polly_getPTXModule";
  llvm::Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(PointerType::getUnqual(getGPUModulePtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, Buffer, Module);
}

void IslPTXGenerator::createCallGetPTXKernelEntry(Value *Entry, Value *Module,
                                                  Value *Kernel) {
  const char *Name = "polly_getPTXKernelEntry";
  llvm::Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(getGPUModulePtrType());
    Args.push_back(PointerType::getUnqual(getGPUFunctionPtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, Entry, Module, Kernel);
}

void IslPTXGenerator::createCallAllocateMemoryForDevice(Value *DeviceData,
                                                        Value *Size) {
  const char *Name = "polly_allocateMemoryForDevice";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(PointerType::getUnqual(getPtrGPUDevicePtrType()));
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, DeviceData, Size);
}

void IslPTXGenerator::createCallCopyFromHostToDevice(Value *DeviceData,
                                                     Value *HostData,
                                                     Value *Size) {
  const char *Name = "polly_copyFromHostToDevice";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(getI8PtrType());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, DeviceData, HostData, Size);
}

void IslPTXGenerator::createCallCopyFromDeviceToHost(Value *HostData,
                                                     Value *DeviceData,
                                                     Value *Size) {
  const char *Name = "polly_copyFromDeviceToHost";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getI8PtrType());
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, HostData, DeviceData, Size);
}

void IslPTXGenerator::createCallSetKernelParameters(Value *Kernel,
                                                    Value *BlockWidth,
                                                    Value *BlockHeight,
                                                    Value *DeviceData,
                                                    Value *ParamOffset) {
  const char *Name = "polly_setKernelParameters";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getGPUFunctionPtrType());
    Args.push_back(getInt64Type());
    Args.push_back(getInt64Type());
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(PointerType::getUnqual(getInt64Type()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall5(F, Kernel, BlockWidth, BlockHeight, DeviceData,
                      ParamOffset);
}

void IslPTXGenerator::createCallLaunchKernel(Value *Kernel, Value *GridWidth,
                                             Value *GridHeight) {
  const char *Name = "polly_launchKernel";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getGPUFunctionPtrType());
    Args.push_back(getInt64Type());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, Kernel, GridWidth, GridHeight);
}

void IslPTXGenerator::createCallStartTimerByCudaEvent(Value *StartEvent,
                                                      Value *StopEvent) {
  const char *Name = "polly_startTimerByCudaEvent";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(PointerType::getUnqual(getGPUEventPtrType()));
    Args.push_back(PointerType::getUnqual(getGPUEventPtrType()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall2(F, StartEvent, StopEvent);
}

void IslPTXGenerator::createCallStopTimerByCudaEvent(Value *StartEvent,
                                                     Value *StopEvent,
                                                     Value *Timer) {
  const char *Name = "polly_stopTimerByCudaEvent";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getGPUEventPtrType());
    Args.push_back(getGPUEventPtrType());
    Args.push_back(getFloatPtrType());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, StartEvent, StopEvent, Timer);
}

void IslPTXGenerator::createCallCleanupGPGPUResources(Value *DeviceData,
                                                      Value *Module,
                                                      Value *Context,
                                                      Value *Kernel,
                                                      Value *Device) {
  const char *Name = "polly_cleanupGPGPUResources";
  llvm::Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(getGPUModulePtrType());
    Args.push_back(getGPUContextPtrType());
    Args.push_back(getGPUFunctionPtrType());
    Args.push_back(getGPUDevicePtrType());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall5(F, DeviceData, Module, Context, Kernel, Device);
}

void IslPTXGenerator::createCallBarrierIntrinsic() {
  Function *Syn =
      Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_barrier0);
  Builder.CreateCall(Syn);
}

void IslPTXGenerator::addKernelSynchronization() {
  createCallBarrierIntrinsic();
}

Value *IslPTXGenerator::getCUDAGridDimX() {
  return GridDimX;
}

Value *IslPTXGenerator::getCUDAGridDimY() {
  if (!GridDimY)
    return ConstantInt::get(getInt64Type(), 1);
  else
    return GridDimY;
}

Value *IslPTXGenerator::getCUDABlockDimX() {
  return ConstantInt::get(getInt64Type(), Kernel->block_dim[0]);
}

Value *IslPTXGenerator::getCUDABlockDimY() {
  if (Kernel->n_block >= 2)
    return ConstantInt::get(getInt64Type(), Kernel->block_dim[1]);
  else
    return ConstantInt::get(getInt64Type(), 1);
}

Value *IslPTXGenerator::getCUDABlockDimZ() {
  if (Kernel->n_block == 3)
    return ConstantInt::get(getInt64Type(), Kernel->block_dim[2]);
  else
    return ConstantInt::get(getInt64Type(), 1);
}

Value *IslPTXGenerator::getOutputArraySizeInBytes() {
  return ConstantInt::get(getInt64Type(), OutputBytes);
}

static Module *extractPTXFunctionsFromModule(const Module *M,
                                             const StringRef &Triple) {
  llvm::ValueToValueMapTy VMap;
  Module *New = new Module("TempGPUModule", M->getContext());
  New->setTargetTriple(Triple::normalize(Triple));

  // Loop over the functions in the module, making external functions as before
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    if (!I->isDeclaration() &&
        (I->getCallingConv() == CallingConv::PTX_Device ||
         I->getCallingConv() == CallingConv::PTX_Kernel)) {
      Function *NF =
          Function::Create(cast<FunctionType>(I->getType()->getElementType()),
                           I->getLinkage(), I->getName(), New);
      NF->copyAttributesFrom(I);
      VMap[I] = NF;

      Function::arg_iterator DestI = NF->arg_begin();
      for (Function::const_arg_iterator J = I->arg_begin(); J != I->arg_end();
           ++J) {
        DestI->setName(J->getName());
        VMap[J] = DestI++;
      }

      SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
      CloneFunctionInto(NF, I, VMap, /*ModuleLevelChanges=*/true, Returns);
    }
  }

  return New;
}

static bool createASMAsString(Module *New, const StringRef &Triple,
                              const StringRef &MCPU, const StringRef &Features,
                              std::string &ASM) {
  llvm::Triple TheTriple(Triple::normalize(Triple));
  std::string ErrMsg;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TheTriple.getTriple(), ErrMsg);

  if (!TheTarget) {
    errs() << ErrMsg << "\n";
    return false;
  }

  TargetOptions Options;
  std::unique_ptr<TargetMachine> target(TheTarget->createTargetMachine(
      TheTriple.getTriple(), MCPU, Features, Options));
  assert(target.get() && "Could not allocate target machine!");
  TargetMachine &Target = *target.get();

  // Build up all of the passes that we want to do to the module.
  PassManager PM;

  TargetLibraryInfo *TLI = new TargetLibraryInfo(TheTriple);
  PM.add(TLI);

  PM.add(new DataLayoutPass(*Target.getDataLayout()));
  Target.addAnalysisPasses(PM);

  {
    raw_string_ostream NameROS(ASM);
    formatted_raw_ostream FOS(NameROS);

    // Ask the target to add backend passes as necessary.
    int UseVerifier = true;
    if (Target.addPassesToEmitFile(PM, FOS, TargetMachine::CGFT_AssemblyFile,
                                   UseVerifier)) {
      errs() << "The target does not support generation of this file type!\n";
      return false;
    }

    PM.run(*New);
    FOS.flush();
  }

  return true;
}

Value *IslPTXGenerator::createPTXKernelFunction(Function *SubFunction) {
  Module *M = getModule();
  Module *GPUModule = extractPTXFunctionsFromModule(M, GPUTriple);
  std::string LLVMKernelStr;
  if (!createASMAsString(GPUModule, GPUTriple, "sm_20" /*MCPU*/,
                         "" /*Features*/, LLVMKernelStr)) {
    errs() << "Generate ptx string failed!\n";
    return nullptr;
  }

  Value *LLVMKernel =
      Builder.CreateGlobalStringPtr(LLVMKernelStr, "llvm_kernel");

  delete GPUModule;
  return LLVMKernel;
}

Value *IslPTXGenerator::getPTXKernelEntryName(Function *SubFunction) {
  StringRef Entry = SubFunction->getName();
  return Builder.CreateGlobalStringPtr(Entry, "ptx_entry");
}

void IslPTXGenerator::eraseUnusedFunctions(Function *SubFunction) {
  Module *M = getModule();
  SubFunction->eraseFromParent();

  if (Function *FuncPTXReadNCtaidX = M->getFunction("llvm.ptx.read.nctaid.x")) {
    FuncPTXReadNCtaidX->eraseFromParent();
  }

  if (Function *FuncPTXReadNCtaidY = M->getFunction("llvm.ptx.read.nctaid.y")) {
    FuncPTXReadNCtaidY->eraseFromParent();
  }

  if (Function *FuncPTXReadCtaidX = M->getFunction("llvm.ptx.read.ctaid.x")) {
    FuncPTXReadCtaidX->eraseFromParent();
  }

  if (Function *FuncPTXReadCtaidY = M->getFunction("llvm.ptx.read.ctaid.y")) {
    FuncPTXReadCtaidY->eraseFromParent();
  }

  if (Function *FuncPTXReadNTidX = M->getFunction("llvm.ptx.read.ntid.x")) {
    FuncPTXReadNTidX->eraseFromParent();
  }

  if (Function *FuncPTXReadNTidY = M->getFunction("llvm.ptx.read.ntid.y")) {
    FuncPTXReadNTidY->eraseFromParent();
  }

  if (Function *FuncPTXReadNTidZ = M->getFunction("llvm.ptx.read.ntid.z")) {
    FuncPTXReadNTidZ->eraseFromParent();
  }

  if (Function *FuncPTXReadTidX = M->getFunction("llvm.ptx.read.tid.x")) {
    FuncPTXReadTidX->eraseFromParent();
  }

  if (Function *FuncPTXReadTidY = M->getFunction("llvm.ptx.read.tid.y")) {
    FuncPTXReadTidY->eraseFromParent();
  }

  if (Function *FuncPTXReadTidZ = M->getFunction("llvm.ptx.read.tid.z")) {
    FuncPTXReadTidZ->eraseFromParent();
  }

  if (Function *FuncNVVMBarrier0 = M->getFunction("llvm.nvvm.barrier0")) {
    FuncNVVMBarrier0->eraseFromParent();
  }
}

static unsigned getArraySizeInBytes(const ArrayType *AT) {
  unsigned Bytes = AT->getNumElements();
  if (const ArrayType *T = dyn_cast<ArrayType>(AT->getElementType()))
    Bytes *= getArraySizeInBytes(T);
  else
    Bytes *= AT->getElementType()->getPrimitiveSizeInBits() / 8;

  return Bytes;
}

void IslPTXGenerator::setLaunchingParameters(Value *GridSizeX,
                                             Value *GridSizeY) {
  GridDimX = GridSizeX;
  GridDimY = GridSizeY;
}

void IslPTXGenerator::finishGeneration(Function *F) {
  // Define data used by the GPURuntime library.
  AllocaInst *PtrCUContext =
      Builder.CreateAlloca(getGPUContextPtrType(), 0, "phcontext");
  AllocaInst *PtrCUDevice =
      Builder.CreateAlloca(getGPUDevicePtrType(), 0, "phdevice");
  AllocaInst *PtrCUModule =
      Builder.CreateAlloca(getGPUModulePtrType(), 0, "phmodule");
  AllocaInst *PtrCUKernel =
      Builder.CreateAlloca(getGPUFunctionPtrType(), 0, "phkernel");
  AllocaInst *PtrCUStartEvent =
      Builder.CreateAlloca(getGPUEventPtrType(), 0, "pstart_timer");
  AllocaInst *PtrCUStopEvent =
      Builder.CreateAlloca(getGPUEventPtrType(), 0, "pstop_timer");
  Type *FloatTy = llvm::Type::getFloatTy(getModule()->getContext());
  AllocaInst *PtrElapsedTimes = Builder.CreateAlloca(FloatTy, 0, "ptimer");
  PtrElapsedTimes->setAlignment(FloatTy->getPrimitiveSizeInBits()/8);
  Builder.CreateStore(ConstantFP::get(FloatTy, 0.0), PtrElapsedTimes);
  AllocaInst *PtrParamOffset =
      Builder.CreateAlloca(getInt64Type(), 0, "pparamoffset");
  PtrParamOffset->setAlignment(8/*Int64 Type*/);
  Builder.CreateStore(ConstantInt::get(getInt64Type(), 0), PtrParamOffset);

  // Initialize the GPU device.
  createCallInitDevice(PtrCUContext, PtrCUDevice);

  // Create the GPU kernel module and entry function.
  Value *PTXString = createPTXKernelFunction(F);
  assert(PTXString && "The generated ptx string should never be empty.");
  Value *PTXEntry = getPTXKernelEntryName(F);
  createCallGetPTXModule(PTXString, PtrCUModule);
  LoadInst *CUModule = Builder.CreateLoad(PtrCUModule, "cumodule");
  createCallGetPTXKernelEntry(PTXEntry, CUModule, PtrCUKernel);

  LoadInst *CUKernel = Builder.CreateLoad(PtrCUKernel, "cukernel");

  // Allocate memory space for output array.
  llvm::ValueToValueMapTy VMap;
  Value *OutputAddr;
  SetVector<Value *> OutAValues;
  SetVector<Value *> InAValues;
  SetVector<Value *> SValues;
  // I should initial all three above Value * vector with correct information
  // initialize(OutAValues, InAValues, SValues);
  //

  for (SetVector<Value *>::iterator I = OutAValues.begin(),
                                    E = OutAValues.end();
       I != E; ++I) {
    Value *BaseAddr = *I;
    OutputAddr = BaseAddr;
    if (const PointerType *PT = dyn_cast<PointerType>(BaseAddr->getType())) {
      Type *T = PT->getArrayElementType();
      const ArrayType *ATy = dyn_cast<ArrayType>(T);
      unsigned Bytes = getArraySizeInBytes(ATy);
      OutputBytes = Bytes;
      std::string PtrDevName("pdevice_");
      PtrDevName.append(BaseAddr->getName().str());
      AllocaInst *PtrDevData =
          Builder.CreateAlloca(getPtrGPUDevicePtrType(), 0, PtrDevName);
      VMap[BaseAddr] = PtrDevData;

      // allocate memory for input array on the device.
      Value *ArraySize = ConstantInt::get(getInt64Type(), Bytes);
      createCallAllocateMemoryForDevice(PtrDevData, ArraySize);

      std::string DevName("device_");
      DevName.append(BaseAddr->getName().str());
      LoadInst *DData = Builder.CreateLoad(PtrDevData, DevName);
      createCallSetKernelParameters(CUKernel, getCUDABlockDimX(),
                                    getCUDABlockDimY(), DData,
                                    PtrParamOffset);

      if (InAValues.count(BaseAddr)) {
        std::string HostName("host_");
        HostName.append(BaseAddr->getName().str());
        Value *HData =
            Builder.CreateBitCast(BaseAddr, getI8PtrType(), HostName);
        createCallCopyFromHostToDevice(DData, HData, ArraySize);
      }
    }
  }

  // Allocate device memory for scalar parameters.
  for (SetVector<Value *>::iterator I = SValues.begin(), E = SValues.end();
       I != E; ++I) {
    Value *SC = *I;
    unsigned Bytes = SC->getType()->getPrimitiveSizeInBits() / 8;
    AllocaInst *PtrDevData =
        Builder.CreateAlloca(getPtrGPUDevicePtrType(), 0, "pdevice_scalar");
    Value *Size = ConstantInt::get(getInt64Type(), Bytes);
    createCallAllocateMemoryForDevice(PtrDevData, Size);

    AllocaInst *TempHData = Builder.CreateAlloca(SC->getType(), 0, "");
    Builder.CreateStore(SC, TempHData);
    Value *HData =
        Builder.CreateBitCast(TempHData, getI8PtrType(), "host_scalar");
    LoadInst *DData = Builder.CreateLoad(PtrDevData, "device_scalar");
    createCallCopyFromHostToDevice(DData, HData, Size);

    createCallSetKernelParameters(CUKernel, getCUDABlockDimX(),
                                  getCUDABlockDimY(), DData, PtrParamOffset);
  }

  // Allocate device memory and its corresponding host memory.
  // We read from Stmt about the info of memory access.
  for (SetVector<Value *>::iterator I = InAValues.begin(), E = InAValues.end();
       I != E; ++I) {
    Value *BaseAddr = *I;
    if (OutAValues.count(BaseAddr))
      continue;

    if (const PointerType *PT = dyn_cast<PointerType>(BaseAddr->getType())) {
      Type *T = PT->getArrayElementType();
      const ArrayType *ATy = dyn_cast<ArrayType>(T);
      unsigned Bytes = getArraySizeInBytes(ATy);
      std::string PtrDevName("pdevice_");
      PtrDevName.append(BaseAddr->getName().str());
      AllocaInst *PtrDevData =
          Builder.CreateAlloca(getPtrGPUDevicePtrType(), 0, PtrDevName);
      VMap[BaseAddr] = PtrDevData;

      // allocate memory for input array on the device.
      Value *ArraySize = ConstantInt::get(getInt64Type(), Bytes);
      createCallAllocateMemoryForDevice(PtrDevData, ArraySize);

      // copy input array to the device
      std::string HostName("host_");
      HostName.append(BaseAddr->getName().str());
      Value *HData = Builder.CreateBitCast(BaseAddr, getI8PtrType(), HostName);
      std::string DevName("device_");
      DevName.append(BaseAddr->getName().str());
      LoadInst *DData = Builder.CreateLoad(PtrDevData, DevName);
      createCallCopyFromHostToDevice(DData, HData, ArraySize);

      // add this parameter to gpu function
      createCallSetKernelParameters(CUKernel, getCUDABlockDimX(),
                                    getCUDABlockDimY(), DData,
                                    PtrParamOffset);
    }
  }

  // Create the start and end timer and record the start time.
  createCallStartTimerByCudaEvent(PtrCUStartEvent, PtrCUStopEvent);

  // Launch the GPU kernel.
  createCallLaunchKernel(CUKernel, getCUDAGridDimX(), getCUDAGridDimY());

  // Copy the results back from the GPU to the host.
  // Value *HData = Builder.CreateBitCast(OutputAddr, getI8PtrType(),
  // "host_data");
  // LoadInst *DData = Builder.CreateLoad(VMap[OutputAddr], "device_data");
  // createCallCopyFromDeviceToHost(HData, DData, getOutputArraySizeInBytes());

  // Record the end time.
  LoadInst *CUStartEvent = Builder.CreateLoad(PtrCUStartEvent, "start_timer");
  LoadInst *CUStopEvent = Builder.CreateLoad(PtrCUStopEvent, "stop_timer");
  createCallStopTimerByCudaEvent(CUStartEvent, CUStopEvent, PtrElapsedTimes);

  // Cleanup all the resources used.
  LoadInst *CUContext = Builder.CreateLoad(PtrCUContext, "cucontext");
  LoadInst *CUDevice = Builder.CreateLoad(PtrCUDevice, "cudevice");
  // createCallCleanupGPGPUResources(DData, CUModule, CUContext, CUKernel,
  //                                CUDevice);

  // Erase the ptx kernel and device subfunctions and ptx intrinsics from
  // current module.
  eraseUnusedFunctions(F);
}
#endif /* GPU_CODEGEN */
