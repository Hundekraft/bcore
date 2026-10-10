#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/CGSCCPassManager.h>

#include "cpu_state.h"
#include "mem.h"

namespace llvm { class TargetMachine; class JITEventListener; }
class DiskObjectCache;
class ObjectPack;

// Function signature for JIT'd basic block functions
using BbFunc = void (*)(CpuState*, Memory*);

struct BBInfo {
    BbFunc fn = nullptr;
    uint32_t fallthrough_pc = 0;
    BBInfo() = default;
    BBInfo(BbFunc f, uint32_t fpc) : fn(f), fallthrough_pc(fpc) {}
};

class BBTranslator;

class JitEngine {
public:
    JitEngine();
    ~JitEngine();

    // opt_level: the IR pipeline, 0=None, 1=O1, 2=O2, 3=O3.
    // codegen_level: the backend on the same scale; <0 follows opt_level.
    bool init(int opt_level = 2, int codegen_level = -1);
    // Optimize a module in-place using the same pipeline level as init().
    // No-op when opt_level is 0. Core::run calls it on every translated module
    // before addModule; before that, only the IR-dump path did, so opt_level
    // never reached executed code.
    void optimize_module(llvm::Module& mod);
    // The layout and triple LLJIT compiles for. A module carrying exactly these
    // passes addIRModule's check; any other non-default layout is an error.
    std::string dataLayout() const { return jit_->getDataLayout().getStringRepresentation(); }
    std::string targetTriple() const { return jit_->getTargetTriple().str(); }
    bool addModule(uint32_t pc, std::unique_ptr<llvm::Module> module,
                   std::unique_ptr<llvm::LLVMContext> ctx,
                   uint32_t fallthrough_pc);
    BbFunc lookup(uint32_t pc);

    // Persistent object cache: compiled blocks are stored as object files in dir, keyed by a SHA-256 of the
    // block's unoptimized bitcode plus the compile settings, LLVM version and host CPU. Empty dir disables it.
    // A key can only match identical input, so a hit never changes behavior, it only skips optimize+codegen.
    void set_object_cache_dir(const std::string& dir);
    // Objects and the byte index live in this pack (owned by Core, shared across invalidate()).
    void set_object_pack(ObjectPack* pack);
    // Load pc's block straight from the pack when guest memory still holds the bytes it was lifted
    // from (no lifting, no bitcode). Returns true and caches the block on success.
    bool try_load_by_bytes(uint32_t pc, Memory* mem);
    // After a translation: remember which bytes the module tagged by tag_cached_module came from.
    void record_block_bytes(uint32_t pc, uint32_t end_pc, uint32_t fallthrough_pc, Memory* mem,
                            const std::string& module_id);
    // Names the module by its cache key. Returns true when the object is already cached; the caller then
    // skips optimize_module (addModule loads the cached object instead of compiling).
    bool tag_cached_module(llvm::Module& mod);
    // ---- Asynchronous compilation (one worker thread) ----
    // A lifted, uncached block: the worker optimizes and compiles it while the host runs the code some
    // other way (op1emu interprets it). All IR optimization and codegen then happens on the worker, so
    // the single-threaded pass pipeline is never used from two threads.
    struct CompileJob {
        uint32_t pc = 0, end_pc = 0, fallthrough_pc = 0;
        uint64_t bytes_hash = 0;    // guest code bytes [pc, end_pc) when lifted
        std::string module_id;
        // context before module: members are destroyed in reverse order, and a Module must go before its context.
        std::unique_ptr<llvm::LLVMContext> context;
        std::unique_ptr<llvm::Module> module;
    };
    struct CompiledBlock {
        uint32_t pc = 0, end_pc = 0, fallthrough_pc = 0;
        uint64_t bytes_hash = 0;
        std::string module_id;
        BbFunc fn = nullptr;        // nullptr: compile failed
    };
    // Lifts pc into job (module, context, end_pc, fallthrough_pc, bytes_hash, module_id); runs on the worker for
    // jobs queued without a module. Returns false when the block cannot be lifted.
    using Lifter = std::function<bool(uint32_t pc, CompileJob& job)>;
    void start_worker(Lifter lifter);
    bool worker_running() const { return worker_.joinable(); }
    bool is_pending(uint32_t pc) const { return pending_.count(pc) != 0; }   // CPU thread only
    void enqueue(CompileJob job);                                            // CPU thread only
    // Blocks until the job for pc has finished (CPU thread); moves it to the front of the queue first.
    void wait_for(uint32_t pc);
    // Moves pc's queued job to the front (the host keeps running it: it is hot).
    void prioritize(uint32_t pc);
    // Hands finished blocks to the CPU thread (removes them from the pending set).
    std::vector<CompiledBlock> take_compiled();
    void install(uint32_t pc, BbFunc fn, uint32_t fallthrough_pc) { cache_[pc] = BBInfo{fn, fallthrough_pc}; }

    uint64_t cache_hits() const { return cache_hits_; }
    const std::string& cache_salt() const { return cache_salt_; }
    uint64_t cache_misses() const { return cache_misses_; }
    void set_executing(uint32_t pc) { executing_bb_pc_ = pc; }
    void clear_executing()          { executing_bb_pc_ = 0; }

    // Register L on the underlying object-linking layer for object load/free
    // notifications (the profiling code map builds itself from these).
    // Behavior-neutral on x86-64 Linux where LLJIT defaults to
    // RTDyldObjectLinkingLayer; returns false (and registers nothing) on any
    // other layer so callers can degrade gracefully.
    bool registerJITEventListener(llvm::JITEventListener& L);
    // Undo registerJITEventListener (no-op on other linking layers).
    void unregisterJITEventListener(llvm::JITEventListener& L);

    // Iterate all cached blocks as fn(host_addr, guest_pc). Addresses come
    // from the symbol lookup, sizes are not tracked here -- use the load
    // listener (via Core::forEachCompiledBlock) when exact sizes are needed.
    template <typename Fn>
    void forEachCompiled(Fn&& fn) const {
        for (const auto& kv : cache_)
            fn(reinterpret_cast<const void*>(kv.second.fn), kv.first);
    }

private:
    std::unique_ptr<DiskObjectCache> objcache_;  // declared before jit_: the compiler holds a pointer to it
    std::string cache_salt_;
    std::atomic<uint64_t> cache_hits_{0}, cache_misses_{0};   // worker and CPU thread
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unordered_map<uint32_t, BBInfo> cache_;
    int opt_level_ = 2;
    uint32_t executing_bb_pc_ = 0;  // PC of currently-executing BB (0 = none)

    // addIRModule + symbol lookup (compiles on the calling thread); does not touch cache_.
    BbFunc materialize(uint32_t pc, std::unique_ptr<llvm::Module> module, std::unique_ptr<llvm::LLVMContext> ctx);
    void worker_loop();
    Lifter lifter_;
    std::thread worker_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_, done_cv_;
    std::deque<CompileJob> queue_;
    std::vector<CompiledBlock> done_;
    std::unordered_set<uint32_t> pending_;   // CPU thread only
    bool stop_ = false;
    // Cached pass pipeline (built once in init(), reused across optimize_module() calls).
    // pass_machine_ outlives pb_, which holds it.
    std::unique_ptr<llvm::TargetMachine> pass_machine_;
    std::unique_ptr<llvm::PassBuilder> pb_;
};
