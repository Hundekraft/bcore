#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "bcore_profile.h"
#include "cpu_state.h"
#include "mem.h"

class JitEngine;
class BBTranslator;
class ObjectPack;

namespace bcore_profile { struct ProfileState; }

class Core {
public:
    Core(CpuState* cpu, Memory* mem);
    ~Core();

    // opt_level: the IR pass pipeline, 0=None, 1=O1, 2=O2, 3=O3.
    // codegen_level: the backend on the same scale (0=None, 1=Less, 2=Default,
    // 3=Aggressive); it follows opt_level when negative.
    bool init(int opt_level = 2, int codegen_level = -1);
    void invalidate();
    bool run(uint32_t pc);
    // Asynchronous compilation: a block that is neither compiled nor cached is lifted here and compiled on a
    // worker thread; run() then returns false with not_ready() set, and the host executes the code some
    // other way (op1emu interprets one instruction) and calls run() again. run_sync() waits for the
    // compile instead (for code the host cannot run otherwise). Off by default (deterministic hosts).
    void set_async_compile(bool on);
    bool run_sync(uint32_t pc);
    bool not_ready() const { return not_ready_; }
    std::tuple<std::string, uint32_t> disassemble(uint32_t pc);
    void set_dump_ir(bool enable);
    // Persistent JIT object cache directory (see JitEngine::set_object_cache_dir); "" disables it.
    // Kept across invalidate(). Call any time; applies to blocks translated from then on.
    void set_object_cache_dir(const std::string& dir);
    // Up to this many packets per run(): a block whose exit jumps back to its own start (hardware-loop
    // iteration, tight polling loop) runs again directly, advancing CYCLES as the host would between
    // blocks. Stops on any other pc, halt, IDLE, a deliverable interrupt, or !Memory::allow_chain().
    // 0 (default) = one block per run().
    void set_chain_budget(uint32_t packets) { chain_budget_ = packets; }
    uint64_t object_cache_hits() const;
    uint64_t object_cache_misses() const;
    // Host time spent translating blocks that were not already loaded (lift + optimize + codegen or cache load).
    uint64_t translate_ns() const { return translate_ns_; }
    uint64_t translate_count() const { return translate_count_; }

    // ---- JIT observability (contract in bcore_profile.h) ----

    // Iterate every currently-compiled block with its host address and exact
    // size. The map is rebuilt by the JIT load listener; after invalidate()
    // only blocks compiled since then appear.
    void forEachCompiledBlock(const std::function<void(const BcoreBlockInfo&)>& fn) const;

    // Attach a typed event sink. Not owned; must outlive the Core or be
    // detached with setEventSink(nullptr) first.
    void setEventSink(BcoreEventSink* sink);

    // Counter snapshot for hosts that want numbers without per-event cost.
    BcoreStats stats() const;

    // Enable before capture. Retain at most capacity unloaded code ranges;
    // live ranges are always available. Loss is explicit, never reattributed.
    void setProfileHistory(size_t capacity);
    void forEachProfileBlock(const std::function<void(const BcoreBlockInfo&)>& fn) const;
    uint64_t profileHistoryDropped() const;

    // False when the object linking layer doesn't support JIT event
    // listeners; the code map then reports host_size == 0 (lookup-derived
    // addresses only).
    bool profileHasBlockSizes() const;

    // Runtime switch for jitdump emission (`perf inject --jit`). Takes effect
    // immediately on the current engine (safe any time after init(),
    // including from a running boot) and persists across invalidate():
    // enabling writes every block compiled from then on, disabling stops
    // that; records already written stay in the dump. Only has an effect in
    // builds with BCORE_PERF_JIT_EVENTS=ON.
    void set_perf_jitdump(bool enable);

private:
    void applyModuleTarget();
    bool run_impl(uint32_t pc, bool may_defer);
    void install_compiled();
    void start_async_worker();
    uint64_t hash_code(uint32_t pc, uint32_t end_pc) const;
    // (Re)registers the code-map listener on jit_. Called by init/invalidate.
    void attachProfiler();

    CpuState* cpu_;
    Memory* mem_;
    // Declared before jit_ on purpose: a JitEngine being destroyed frees its
    // objects through the registered listener, which writes into profile_ --
    // so profile_ must outlive any engine.
    std::unique_ptr<bcore_profile::ProfileState> profile_;
    std::shared_ptr<JitEngine> jit_;
    std::shared_ptr<BBTranslator> translator_;
    int opt_level_ = 2;
    std::string object_cache_dir_;
    std::shared_ptr<ObjectPack> pack_;
    std::string translation_salt_;
    uint32_t chain_budget_ = 0;
    uint64_t cache_hits_base_ = 0, cache_misses_base_ = 0;
    uint64_t translate_ns_ = 0, translate_count_ = 0;
    int codegen_level_ = -1;
    bool dump_ir_ = false;
    bool async_ = false;
    bool not_ready_ = false;
    std::unordered_map<uint32_t, uint32_t> pending_hits_;   // async: times the host arrived at a compiling block
};
