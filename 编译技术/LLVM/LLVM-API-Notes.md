## 主要类的用法

* Module contains Functions/GlobalVariables
  * Module is unit of compilation/analysis/optimization
* Function contains BasicBlocks/Arguments
  * Functions roughly correspond to functions in C
* BasicBlock contains list of instructions
  * Each block ends in a control flow instruction
* Instruction is opcode + vector of operands
  * All operands have types
  * Instruction result is typed

## Pass

* Compiler is organized as a series of ‘passes’:
  * Each pass is one analysis or transformation
* Four types of Pass:
  * ModulePass: general interprocedural pass
  * CallGraphSCCPass: bottom-up on the call graph
  * FunctionPass: process a function at a time
  * BasicBlockPass: process a basic block at a time
* `PassManager` provides:
  * Process a function at a time instead of a pass at a time
  * Declarative dependency management:
    * Share analyses between passes when safe

````
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();        // Get aliases
      AU.addRequired<TargetData>();           // Get data layout
      CallGraphSCCPass::getAnalysisUsage(AU); // Get CallGraph
    }
````

## IR

* LLVM IR is in SSA form:
  * use-def and def-use chains are always available
  * All objects have user/use info, even functions
* CFG is always available:
  * Exposed as BasicBlock predecessor/successor lists
* Higher-level info implemented as passes:
  * Dominators, CallGraph, induction vars, aliasing, GVN

## Misc

Some simple op on IR: [doc](https://llvm.org/docs/ProgrammersManual.html#helpful-hints-for-common-operations)

## 遍历

````cpp
// 遍历函数里的基本块
Function &Func = ...
for (BasicBlock &BB : Func)
  // Print out the name of the basic block if it has one, and then the
  // number of instructions that it contains
  errs() << "Basic block (name=" << BB.getName() << ") has "
             << BB.size() << " instructions.\n";


// 遍历基本块里的指令
BasicBlock& BB = ...
for (Instruction &I : BB)
   // The next statement works since operator<<(ostream&,...)
   // is overloaded for Instruction&
   errs() << I << "\n";


// 遍历函数里的指令
// F is a pointer to a Function instance
for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
  errs() << *I << "\n";


// 类型判断
// 寻找所有对 targetFunc 的调用
Function* targetFunc = ...;

class OurFunctionPass : public FunctionPass {
  public:
    OurFunctionPass(): callCounter(0) { }

    virtual runOnFunction(Function& F) {
      for (BasicBlock &B : F) {
        for (Instruction &I: B) {
          if (auto *CB = dyn_cast<CallBase>(&I)) {
            // We know we've encountered some kind of call instruction (call,
            // invoke, or callbr), so we need to determine if it's a call to
            // the function pointed to by m_func or not.
            if (CB->getCalledFunction() == targetFunc)
              ++callCounter;
          }
        }
      }
    }

  private:
    unsigned callCounter;
};


// Use-Def 链
Function *F = ...;

for (User *U : F->users()) {
  if (Instruction *Inst = dyn_cast<Instruction>(U)) {
    errs() << "F is used in instruction:\n";
    errs() << *Inst << "\n";
  }


// BasicBlock 的前后继
#include "llvm/IR/CFG.h"
BasicBlock *BB = ...;

for (BasicBlock *Pred : predecessors(BB)) {
  // ...
}
````

## 简单修改

````cpp
auto *ai = new AllocaInst(Type::Int32Ty);


// 结果变量带名字的
auto *pa = new AllocaInst(Type::Int32Ty, 0, "indexLoc");


// 插入指令到基本块指定位置:
BasicBlock *pb = ...;
Instruction *pi = ...;
auto *newInst = new Instruction(...);

pb->getInstList().insert(pi, newInst); // Inserts newInst before pi in pb


// 附加到末尾:
BasicBlock *pb = ...;
auto *newInst = new Instruction(...);

pb->getInstList().push_back(newInst); // Appends newInst to pb
// 或者直接这样:
BasicBlock *pb = ...;
auto *newInst = new Instruction(..., pb);


// 使用 IRBuilder

````

Insert/Remove/Move/Replace Instructions 
• Three Options 
•Instruction class methods: 
• insertBefore(), insertAfter(), moveBefore(), moveAfter(), eraseFromParent(), removeFromParent(), … 
• Ask parent (BasicBlock) to do this: 
• inst.getParent()->getInstList() .insert/erase/remove/…() 
• Make use of BasicBlockUtils (defined in header llvm/Transforms/Utils/BasicBlockUtils.h): 
• ReplaceInstWithValue(), ReplaceInstwithInst()

## 类

[Value](https://llvm.org/doxygen/classllvm_1_1Value.html)

Is Instruction (User) a Value (Usee)? 
`%2 = add %1, 10`
• DO NOT interpret this statement as “the result of Instruction add %1, 10 is assigned to %2”, instead, think this way – “%2 is the Value Representation of Instruction add %1, 10”. 
• Therefore, whenever we use the Value %2, we mean to use the Instruction add %1, 10.
