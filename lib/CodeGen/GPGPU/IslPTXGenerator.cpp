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
#include "llvm/IR/Verifier.h"
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
  initializeBaseAddresses();
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

void IslPTXGenerator::initializeBaseAddresses() {
  polly::Scop *S = getPollyScop();
  for (ScopStmt *Stmt : *S)
    for (MemoryAccess *Acc : *Stmt)
      BaseAddresses.insert(const_cast<Value *>(Acc->getBaseAddr()));
}

static Type *getArrayElementType(Type *Ty) {
  ArrayType *AT;
  while (AT = dyn_cast<ArrayType>(Ty)) {
    Ty = AT->getElementType();
  }

  return Ty;
}

/* The arguments are placed in the following order:
 * - the arrays accessed by the kernel
 * - the parameters
 * - the host loop iterators
*/
Function *IslPTXGenerator::createSubfunctionDefinition(
    int &NumMemAccs, int &NumVars, int &NumHostIters,
    SmallVector<isl_id *, 4> &ArrayIDs) {
  assert(Kernel && "Kernel should have been set correctly.");

  isl_space *Space;
  for (int i = 0; i < Prog->n_array; ++i) {
    Space = isl_space_copy(Prog->array[i].space);
    isl_set *Arr = isl_union_set_extract_set(Kernel->arrays, Space);
    int Empty = isl_set_plain_is_empty(Arr);
    if (!Empty) {
      isl_id *Id = isl_set_get_tuple_id(Arr);
      ArrayIDs.push_back(Id);
      NumMemAccs++;

      isl_id_free(Id);
    }

    isl_set_free(Arr);
  }

  Space = isl_union_set_get_space(Kernel->arrays);
  NumVars = isl_space_dim(Space, isl_dim_param);
  NumHostIters = isl_space_dim(Kernel->space, isl_dim_set);

  Module *M = getModule();
  Function *F = Builder.GetInsertBlock()->getParent();
  std::vector<Type *> Arguments;

  for (int i = 0; i < NumMemAccs + NumVars + NumHostIters; i++)
    Arguments.push_back(Builder.getInt8PtrTy());

  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *FN = Function::Create(FT, Function::InternalLinkage,
                                  F->getName() + "_ptx_subfn", M);
  FN->setCallingConv(CallingConv::PTX_Kernel);

  // Do not run any optimization pass on the new function.
  P->getAnalysis<polly::ScopDetection>().markFunctionAsInvalid(FN);

  int j = 0;
  for (Function::arg_iterator AI = FN->arg_begin(); AI != FN->arg_end(); ++AI) {
    if (j < NumMemAccs)
      AI->setName("ptx.Array");
    else if (j < NumMemAccs + NumVars)
      AI->setName("ptx.Var");
    else
      AI->setName("ptx.HostIter");

    j++;
  }

  isl_space_free(Space);
  return FN;
}

void IslPTXGenerator::buildScop() {
  polly::Scop *S = getPollyScop();
  Scop = ppcg_scop_from_polly_scop(S, Options);
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

  isl_options_set_schedule_outer_coincidence(Ctx, 1);
  isl_options_set_schedule_maximize_band_depth(Ctx, 1);

  if (generate_gpu(Ctx, Scop, Options, &ext) < 0) {
    errs() << "GPGPU code geneation failed.\n";
    return;
  }
  Guard = ext.guard;
  Tree = ext.tree;
  Prog = ext.prog;
}

void IslPTXGenerator::createSubfunction(IDToValueTy &IDToValue,
                                        Function **Subfunction) {
  int NumMemAccs = 0, NumVars = 0, NumHostIters = 0;
  SmallVector<isl_id *, 4> ArrayIDs;
  Function *F = createSubfunctionDefinition(NumMemAccs, NumVars, NumHostIters,
                                            ArrayIDs);

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

  std::map<std::string, Value *> GPUIDMap;
  switch (NumGrid) {
  case 1: {
    Value *BIDx = Builder.CreateCall(GetCtaidX);
    BIDx = Builder.CreateIntCast(BIDx, Ty, false);
    GPUIDMap["b0"] = cast<Value>(BIDx);
    break;
  }
  case 2: {
    Value *BIDx = Builder.CreateCall(GetCtaidX);
    BIDx = Builder.CreateIntCast(BIDx, Ty, false);
    Value *BIDy = Builder.CreateCall(GetCtaidY);
    BIDy = Builder.CreateIntCast(BIDy, Ty, false);
    GPUIDMap["b0"] = cast<Value>(BIDx);
    GPUIDMap["b1"] = cast<Value>(BIDy);
    break;
  }
  default:
    llvm_unreachable("Set grid id error");
  }

  switch (NumBlock) {
  case 1: {
    Value *TIDx = Builder.CreateCall(GetTidX);
    TIDx = Builder.CreateIntCast(TIDx, Ty, false);
    GPUIDMap["t0"] = cast<Value>(TIDx);
    break;
  }
  case 2: {
    Value *TIDx = Builder.CreateCall(GetTidX);
    TIDx = Builder.CreateIntCast(TIDx, Ty, false);
    Value *TIDy = Builder.CreateCall(GetTidY);
    TIDy = Builder.CreateIntCast(TIDy, Ty, false);
    GPUIDMap["t0"] = cast<Value>(TIDx);
    GPUIDMap["t1"] = cast<Value>(TIDy);
    break;
  }
  case 3: {
    Value *TIDx = Builder.CreateCall(GetTidX);
    TIDx = Builder.CreateIntCast(TIDx, Ty, false);
    Value *TIDy = Builder.CreateCall(GetTidY);
    TIDy = Builder.CreateIntCast(TIDy, Ty, false);
    Value *TIDz = Builder.CreateCall(GetTidZ);
    TIDz = Builder.CreateIntCast(TIDz, Ty, false);
    GPUIDMap["t0"] = cast<Value>(TIDx);
    GPUIDMap["t1"] = cast<Value>(TIDy);
    GPUIDMap["t2"] = cast<Value>(TIDz);
    break;
  }
  default:
    llvm_unreachable("Set thread id error");
  }

  // We should fill the IDToValue before this create(node) call.
  for (int i = 0; i < Kernel->n_gpuid; ++i) {
    isl_id *GPUId = isl_id_copy(Kernel->gpuid[i]);
    std::string Name = isl_id_get_name(GPUId);
    Value *IDValue = GPUIDMap[Name];
    IDToValue[GPUId] = IDValue;

    isl_id_free(GPUId);
  }

  isl_space *Space = isl_union_set_get_space(Kernel->arrays);
  int j = 0;
  for (Function::arg_iterator AI = F->arg_begin(); AI != F->arg_end(); ++AI) {
    if (j < NumMemAccs) {
      isl_id *Id = ArrayIDs[j];
      Value *BaseAddr = (Value *)isl_id_get_user(Id);
      Type *ArrayTy = BaseAddr->getType();
      Type *EleTy =
          getArrayElementType(cast<PointerType>(ArrayTy)->getElementType());
      Type *PointerToEleTy = PointerType::get(EleTy, 1);
      Value *Param = Builder.CreateBitCast(AI, PointerToEleTy);
      IDToValue[Id] = Param;
    } else if (j < NumMemAccs + NumVars) {
      for (int i = 0; i < NumVars; ++i) {
        isl_id *Id = isl_space_get_dim_id(Space, isl_dim_param, i);
        isl_id_free(Id);
      }
    } else {
      // type = isl_options_get_ast_iterator_type(Prog->ctx);
      for (int i = 0; i < NumHostIters; ++i) {
        isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
        IDToValue.erase(IDToValue.find(Id));
        const char *Name =
            isl_space_get_dim_name(Kernel->space, isl_dim_set, i);
        std::string HIName("device_");
        HIName.append(Name);
        LoadInst *HI = Builder.CreateLoad(AI, HIName);
        IDToValue[Id] = HI;

        isl_id_free(Id);
      }
    }

    ++j;
  }
  isl_space_free(Space);
  ArrayIDs.clear();

  Builder.CreateBr(BodyBB);
  Builder.SetInsertPoint(BodyBB);

  // Create the exit block.
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

void IslPTXGenerator::startGeneration(struct ppcg_kernel *CurKernel,
                                      IDToValueTy &IDToValue,
                                      BasicBlock::iterator *KernelBody) {
  Function *SubFunction;
  BasicBlock::iterator PrevInsertPoint = Builder.GetInsertPoint();
  Kernel = CurKernel;
  createSubfunction(IDToValue, &SubFunction);
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
    Args.push_back(getPtrGPUDevicePtrType());
    Args.push_back(PointerType::getUnqual(getInt64Type()));
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall3(F, Kernel, DeviceData, ParamOffset);
}

void IslPTXGenerator::createCallSetBlockShape(Value *Kernel, Value *BlockWidth,
                                              Value *BlockHeight,
                                              Value *BlockDepth) {
  const char *Name = "polly_setBlockShape";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(getGPUFunctionPtrType());
    Args.push_back(getInt64Type());
    Args.push_back(getInt64Type());
    Args.push_back(getInt64Type());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall4(F, Kernel, BlockWidth, BlockHeight, BlockDepth);
}

Value *IslPTXGenerator::getBaseAddressByName(std::string Name) {
  for (Value *Addr : BaseAddresses) {
    std::string AddrName = "MemRef_" + Addr->getName().str();
    if (!strcmp(Name.c_str(), AddrName.c_str()))
      return Addr;
  }

  return nullptr;
}

Value *IslPTXGenerator::getArraySize(struct gpu_array_info *Array,
                                     __isl_take isl_set *Context) {
  Value *ArraySize = ConstantInt::get(getInt64Type(), 1);

  isl_ast_build *Build = isl_ast_build_from_context(Context);
  if (!gpu_array_is_scalar(Array)) {
    isl_ast_expr *Res =
        isl_ast_build_expr_from_pw_aff(Build, isl_pw_aff_copy(Array->bound[0]));
    for (int i = 1; i < Array->n_index; i++) {
      isl_pw_aff *Bound_I = isl_pw_aff_copy(Array->bound[i]);
      isl_ast_expr *Expr = isl_ast_build_expr_from_pw_aff(Build, Bound_I);
      Res = isl_ast_expr_mul(Res, Expr);
    }
    Value *Bound = ExprBuilder.create(Res);
    ArraySize = Builder.CreateMul(ArraySize, Bound);
  }

  isl_ast_build_free(Build);
  return ArraySize;
}

/* This function is duplicated from ppcg/cuda.c and modified. */
void IslPTXGenerator::allocateDeviceArrays(Value *CUKernel,
                                           AllocaInst *PtrParamOffset,
                                           ValueToValueMapTy &VMap) {
  for (int i = 0; i < Prog->n_array; ++i) {
    if (gpu_array_is_read_only_scalar(&Prog->array[i]))
      continue;

    std::string ArrayName = Prog->array[i].name;
    std::string PtrDevName("pdevice_");
    PtrDevName.append(ArrayName);
    AllocaInst *PtrDevData =
        Builder.CreateAlloca(getPtrGPUDevicePtrType(), 0, PtrDevName);

    // allocate memory for input array on the device.
    Value *ArraySize =
        getArraySize(&Prog->array[i], isl_set_copy(Prog->context));
    createCallAllocateMemoryForDevice(PtrDevData, ArraySize);

    std::string DevName("device_");
    DevName.append(ArrayName);
    LoadInst *DData = Builder.CreateLoad(PtrDevData, DevName);
    createCallSetKernelParameters(CUKernel, DData, PtrParamOffset);
    Value *BaseAddr = getBaseAddressByName(ArrayName);
    VMap[BaseAddr] = DData;
  }
}

/* This function is duplicated from ppcg/cuda.c and modified. */
void IslPTXGenerator::copyArraysToDevice(ValueToValueMapTy &VMap) {
  int i;

  for (i = 0; i < Prog->n_array; ++i) {
    isl_space *dim;
    isl_set *read_i;
    int empty;

    if (gpu_array_is_read_only_scalar(&Prog->array[i]))
      continue;

    dim = isl_space_copy(Prog->array[i].space);
    read_i = isl_union_set_extract_set(Prog->copy_in, dim);
    empty = isl_set_plain_is_empty(read_i);
    isl_set_free(read_i);
    if (empty)
      continue;

    std::string ArrayName = Prog->array[i].name;
    Value *BaseAddr = getBaseAddressByName(ArrayName);
    std::string HostName("host_");
    HostName.append(ArrayName);
    Value *HData = nullptr;
    if (gpu_array_is_scalar(&Prog->array[i])) {
      AllocaInst *TempHData = Builder.CreateAlloca(BaseAddr->getType(), 0, "");
      Builder.CreateStore(BaseAddr, TempHData);
      HData = Builder.CreateBitCast(TempHData, getI8PtrType(), "host_scalar");
      VMap[VMap[BaseAddr]] = HData;
    } else
      HData = Builder.CreateBitCast(BaseAddr, getI8PtrType(), HostName);

    Value *ArraySize =
        getArraySize(&Prog->array[i], isl_set_copy(Prog->context));
    createCallCopyFromHostToDevice(VMap[BaseAddr], HData, ArraySize);
  }
}

/* This function is duplicated from ppcg/cuda.c and modified.
 * For each array that needs to be copied out (based on prog->copy_out),
 * copy the contents back from the GPU to the host.
 *
 * If any element of a given array appears in prog->copy_out, then its
 * entire extent is in prog->copy_out.  The bounds on this extent have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
void IslPTXGenerator::copyArraysFromDevice(ValueToValueMapTy VMap) {
  int i;

  for (i = 0; i < Prog->n_array; ++i) {
    isl_space *dim;
    isl_set *copy_out_i;
    int empty;

    dim = isl_space_copy(Prog->array[i].space);
    copy_out_i = isl_union_set_extract_set(Prog->copy_out, dim);
    empty = isl_set_plain_is_empty(copy_out_i);
    isl_set_free(copy_out_i);
    if (empty)
      continue;

    std::string ArrayName = Prog->array[i].name;
    Value *BaseAddr = getBaseAddressByName(ArrayName);
    Value *ArraySize =
        getArraySize(&Prog->array[i], isl_set_copy(Prog->context));

    if (!VMap[VMap[BaseAddr]]) {
      std::string HostName("host_");
      HostName.append(ArrayName);
      Value *HData = Builder.CreateBitCast(BaseAddr, getI8PtrType(), HostName);
      VMap[VMap[BaseAddr]] = HData;
    }

    createCallCopyFromDeviceToHost(VMap[VMap[BaseAddr]], VMap[BaseAddr],
                                   ArraySize);
  }
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

Value *IslPTXGenerator::getCUDAGridDimX() { return GridDimX; }

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

Value *IslPTXGenerator::getGridSize(int Pos) {
  isl_ast_build *Context =
      isl_ast_build_from_context(isl_set_copy(Kernel->context));
  isl_multi_pw_aff *GSize = isl_multi_pw_aff_copy(Kernel->grid_size);
  isl_pw_aff *Size = isl_multi_pw_aff_get_pw_aff(GSize, Pos);
  isl_ast_expr *GridSize = isl_ast_build_expr_from_pw_aff(Context, Size);
  Value *Res = ExprBuilder.create(GridSize);
  isl_multi_pw_aff_free(GSize);
  isl_ast_build_free(Context);

  return Res;
}

void IslPTXGenerator::setLaunchingParameters() {
  int Dim = isl_multi_pw_aff_dim(Kernel->grid_size, isl_dim_set);
  assert((Dim >= 1 && Dim <= 2) && "CUDA grid size should be 1d or 2d.");
  GridDimX = getGridSize(0);
  GridDimY = nullptr;
  if (Dim == 2)
    GridDimY = getGridSize(1);
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
  PtrElapsedTimes->setAlignment(FloatTy->getPrimitiveSizeInBits() / 8);
  Builder.CreateStore(ConstantFP::get(FloatTy, 0.0), PtrElapsedTimes);
  AllocaInst *PtrParamOffset =
      Builder.CreateAlloca(getInt64Type(), 0, "pparamoffset");
  PtrParamOffset->setAlignment(8 /*Int64 Type*/);
  Builder.CreateStore(ConstantInt::get(getInt64Type(), 0), PtrParamOffset);

  // Initialize the GPU device.
  createCallInitDevice(PtrCUContext, PtrCUDevice);

  // Make sure the generated bitcode in subfunction is valid.
  PassManager Passes;
  Passes.add(createVerifierPass());
  Passes.run(*(F->getParent()));

  // Create the GPU kernel module and entry function.
  Value *PTXString = createPTXKernelFunction(F);
  assert(PTXString && "The generated ptx string should never be empty.");
  Value *PTXEntry = getPTXKernelEntryName(F);
  createCallGetPTXModule(PTXString, PtrCUModule);
  LoadInst *CUModule = Builder.CreateLoad(PtrCUModule, "cumodule");
  createCallGetPTXKernelEntry(PTXEntry, CUModule, PtrCUKernel);

  LoadInst *CUKernel = Builder.CreateLoad(PtrCUKernel, "cukernel");

  // Allocate device memory space for copy-in arrays.
  ValueToValueMap VMap;
  allocateDeviceArrays(CUKernel, PtrParamOffset, VMap);

  // Copy the results back from the GPU to the host.
  copyArraysToDevice(VMap);

  // Create the start and end timer and record the start time.
  createCallStartTimerByCudaEvent(PtrCUStartEvent, PtrCUStopEvent);

  // Set kernel block shape.
  createCallSetBlockShape(CUKernel, getCUDABlockDimX(), getCUDABlockDimY(),
                          getCUDABlockDimZ());

  // Launch the GPU kernel.
  setLaunchingParameters();
  createCallLaunchKernel(CUKernel, getCUDAGridDimX(), getCUDAGridDimY());

  // Copy the results back from the GPU to the host.
  copyArraysFromDevice(VMap);

  // Record the end time.
  LoadInst *CUStartEvent = Builder.CreateLoad(PtrCUStartEvent, "start_timer");
  LoadInst *CUStopEvent = Builder.CreateLoad(PtrCUStopEvent, "stop_timer");
  createCallStopTimerByCudaEvent(CUStartEvent, CUStopEvent, PtrElapsedTimes);

  // Erase the ptx kernel and device subfunctions and ptx intrinsics from
  // current module.
  eraseUnusedFunctions(F);
}
#endif /* GPU_CODEGEN */
