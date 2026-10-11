#include <llvm/Support/xxhash.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include "core.h"
#include "mmr.h"

#include <sstream>

#include "bb_translator.h"
#include "decoder.h"
#include "disasm_visitor.h"
#include "jit_engine.h"
#include "object_pack.h"
#include "profile_state.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

Core::Core(CpuState* cpu, Memory* mem)
    : cpu_(cpu), mem_(mem) {
#if BCORE_ENABLE_PROFILE
    profile_ = std::make_unique<bcore_profile::ProfileState>();
#endif
}

Core::~Core() {
    // The JIT engine's compile worker uses translator_ and pack_: stop it (by destroying the engine) before
    // members declared after jit_ go away.
    jit_.reset();
}

void Core::attachProfiler() {
#if BCORE_ENABLE_PROFILE
    profile_->listener_ok = bcore_profile::attach_listeners(*profile_, *jit_);
#endif
}

bool Core::init(int opt_level, int codegen_level) {
    opt_level_ = opt_level;
    codegen_level_ = codegen_level;

    jit_ = std::make_shared<JitEngine>();
    jit_->set_object_cache_dir(object_cache_dir_);
    jit_->set_object_pack(pack_.get());
    if (!jit_->init(opt_level_, codegen_level_))
        return false;
    attachProfiler();

    // Use the Memory interface's is_fastmem() to determine mode.
    const bool fastmem = mem_->is_fastmem();
    // In fastmem mode, fast_base is the mmap base; JIT uses direct loads.
    // In non-fastmem mode, fast_base = data_.data() - base_; pass it for rawmem inline loads.
    const uint64_t fast_base = mem_->fast_base();

    translator_ = std::make_shared<BBTranslator>(
        mem_,
        /*unlimited=*/(cpu_->steps_remaining == 0),
        fastmem,
        fast_base);
    if (cpu_->steps_remaining)
        translator_->set_max_packets(cpu_->steps_remaining);
    applyModuleTarget();

    return true;
}

void Core::applyModuleTarget() {
    // Same source as the compiler: LLJIT's addIRModule rejects a module whose
    // non-default layout differs from its own.
    translator_->setModuleTarget(jit_->dataLayout(), jit_->targetTriple());
}

void Core::set_async_compile(bool on) {
    async_ = on;
    if (on && jit_) start_async_worker();
}

// The worker lifts too: the translator is only used from the worker while async compile is on (and from
// the CPU thread before it starts). Guest code bytes are read from the worker; R246 does not rewrite code
// in place, and install_compiled() drops a block whose bytes changed since the lift.
void Core::start_async_worker() {
    const char* env = getenv("BCORE_COMPILE_THREADS");
    const unsigned threads = env ? (unsigned)strtoul(env, nullptr, 10) : 3;
    jit_->start_worker([this](uint32_t pc, JitEngine::CompileJob& job) {
        // The CPU may be writing this code right now (the boot loader copies stages in while running):
        // a block is only valid when its bytes were the same before and after the lift.
        const uint64_t before = hash_code(pc, pc + 2048);
        BBTranslateResult result;
        {
            std::lock_guard<std::mutex> lock(lift_mu_);   // one translator, several workers
            result = translator_->translate(pc);
        }
        if (!result.module) return false;
        if (hash_code(pc, pc + 2048) != before) return false;
        job.end_pc = result.end_pc;
        job.fallthrough_pc = result.fallthrough_pc;
        job.bytes_hash = hash_code(pc, result.end_pc);
        job.context = std::move(result.context);
        job.module = std::move(result.module);
        return true;
    }, threads);
}

uint64_t Core::hash_code(uint32_t pc, uint32_t end_pc) const {
    std::string bytes;
    if (end_pc > pc) append_guest_bytes(mem_, pc, end_pc - pc, bytes);
    return llvm::xxHash64(bytes);
}

// Installs blocks the worker finished, unless their guest code changed since they were lifted.
void Core::install_compiled() {
    for (auto& d : jit_->take_compiled()) {
        pending_hits_.erase(d.pc);
        if (!d.fn || hash_code(d.pc, d.end_pc) != d.bytes_hash) continue;
        jit_->install(d.pc, d.fn, d.fallthrough_pc);
        jit_->record_block_bytes(d.pc, d.end_pc, d.fallthrough_pc, mem_, d.module_id);
    }
}

bool Core::run(uint32_t pc) { return run_impl(pc, async_); }
bool Core::run_sync(uint32_t pc) { return run_impl(pc, false); }

bool Core::run_impl(uint32_t pc, bool may_defer) {
    not_ready_ = false;
    if (async_) install_compiled();
    BbFunc fn = jit_->lookup(pc);
    if (!fn && async_ && may_defer && jit_->is_pending(pc)) {   // fast path: still compiling
        // A block the host keeps arriving at is hot: compile it next.
        const uint32_t hits = ++pending_hits_[pc];
        if ((hits & (hits - 1)) == 0 && hits >= 2) jit_->prioritize(pc);
        not_ready_ = true;
        return false;
    }
    if (!fn && !async_ && jit_->try_load_by_bytes(pc, mem_))
        fn = jit_->lookup(pc);
    // Async: pack loads go to the workers too (linking an object costs ~0.1-1 ms, many of them in a row
    // stalled the CPU thread when a new screen came up). A failed pack load is retried as a compile.
    for (int attempt = 0; !fn && async_ && attempt < 2; ++attempt) {
        if (!jit_->is_pending(pc)) {
            JitEngine::CompileJob job;   // no module: the worker lifts it (see start_async_worker)
            job.pc = pc;
            if (jit_->match_by_bytes(pc, mem_, job))
                job.bytes_hash = hash_code(pc, job.end_pc);
            else
                ++translate_count_;
            jit_->enqueue(std::move(job));
        }
        if (may_defer) {
            // A block the host keeps arriving at is hot: compile it next.
            const uint32_t hits = ++pending_hits_[pc];
            if ((hits & (hits - 1)) == 0 && hits >= 2) jit_->prioritize(pc);
            not_ready_ = true;
            return false;
        }
        jit_->wait_for(pc);
        install_compiled();
        fn = jit_->lookup(pc);
    }
    if (!fn && async_) return false;
    if (!fn) {
        const auto host_t0 = std::chrono::steady_clock::now();
        struct AddTime {
            uint64_t& ns; uint64_t& n; std::chrono::steady_clock::time_point t0;
            uint32_t pc;
            uint64_t slow_lift_ns = 0, slow_opt_ns = 0;
            unsigned slow_ir = 0, slow_bytes = 0;
            ~AddTime() {
                const uint64_t d = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                ns += d; ++n;
                // BCORE_SLOW_MS=N: report translations slower than N ms (pc, time) on stderr.
                static const long slow = getenv("BCORE_SLOW_MS") ? strtol(getenv("BCORE_SLOW_MS"), nullptr, 10) : -1;
                if (slow >= 0 && d >= (uint64_t)slow * 1000000ull)
                    fprintf(stderr, "bcore: slow translate pc=0x%08x %.1f ms (lift %.1f, optimize %.1f, %u IR insns, %u guest bytes)\n",
                            pc, d / 1e6, slow_lift_ns / 1e6, slow_opt_ns / 1e6, slow_ir, slow_bytes);
            }
        } add_time{translate_ns_, translate_count_, host_t0, pc};
#if BCORE_ENABLE_PROFILE
        BcoreEventSink* sink = profile_->sink;
        profile_->stats.cache_misses++;
        if (sink) sink->onTranslateBegin(pc);
        const uint64_t t0 = bcore_profile::monotonic_ns();
        // Balances onTranslateBegin when anything below throws (a host
        // Memory may: decoding reads guest memory through it). The exception
        // still propagates.
        struct FailOnUnwind {
            BcoreEventSink* sink;
            uint32_t pc;
            uint64_t t0;
            bool armed = true;
            ~FailOnUnwind() {
                if (armed && sink) sink->onTranslateFailure(pc, bcore_profile::monotonic_ns() - t0);
            }
        } fail_on_unwind{sink, pc, t0};
#endif
        auto result = translator_->translate(pc);
        const auto host_lift_end = std::chrono::steady_clock::now();
        add_time.slow_lift_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(host_lift_end - host_t0).count();
        add_time.slow_ir = result.module ? (unsigned)result.module->getInstructionCount() : 0;
        add_time.slow_bytes = result.end_pc > pc ? result.end_pc - pc : 0;
#if BCORE_ENABLE_PROFILE
        // Stage callbacks fire only after the translation is timed, so
        // nothing the sink does lands inside a stage or in translate_ns.
        const uint64_t lift_end = bcore_profile::monotonic_ns();
        uint64_t optimize_end = 0;
#endif
        // The IR pipeline, once per module and before any dump, so a dump shows
        // the IR that is compiled. It used to run only inside the dump branch:
        // from the initial commit on, a nonzero opt_level changed the printed IR
        // and never the executed code.
        // A cached object skips the IR pipeline and codegen: addModule loads it.
        const bool cached = jit_->tag_cached_module(*result.module);
        if (opt_level_ != 0 && !cached) {
            const auto opt_t0 = std::chrono::steady_clock::now();
            jit_->optimize_module(*result.module);
            add_time.slow_opt_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - opt_t0).count();
#if BCORE_ENABLE_PROFILE
            optimize_end = bcore_profile::monotonic_ns();
#endif
        }
        if (dump_ir_) {
            llvm::errs() << "=== IR BB @ " << llvm::format_hex(pc, 10) << " ===\n";
            result.module->print(llvm::errs(), nullptr);
            llvm::errs() << "=== end IR ===\n";
        }
#if BCORE_ENABLE_PROFILE
        const uint64_t materialize_begin = bcore_profile::monotonic_ns();
#endif
        const std::string module_id = result.module->getModuleIdentifier();
        if (jit_->addModule(pc,
                            std::move(result.module),
                            std::move(result.context),
                            result.fallthrough_pc)) {
            fn = jit_->lookup(pc);
            if (fn) jit_->record_block_bytes(pc, result.end_pc, result.fallthrough_pc, mem_, module_id);
        }
#if BCORE_ENABLE_PROFILE
        const uint64_t t1 = bcore_profile::monotonic_ns();
        fail_on_unwind.armed = false;
        profile_->stats.translate_ns_total += t1 - t0;
        if (fn) profile_->stats.blocks_translated++;
        if (sink) {
            sink->onCompileStage(pc, "lift", t0, lift_end, true);
            if (optimize_end) sink->onCompileStage(pc, "ir-optimize", lift_end, optimize_end, true);
            sink->onCompileStage(pc, "materialize", materialize_begin, t1, fn != nullptr);
            if (fn) sink->onTranslateEnd(pc, t1 - t0);
            else sink->onTranslateFailure(pc, t1 - t0);
        }
#endif
        if (!fn)
            return false;
    }
#if BCORE_ENABLE_PROFILE
    else {
        profile_->stats.cache_hits++;
    }
#endif

    cpu_->did_jump = false;
    cpu_->pc = pc;
    cpu_->packets = 0;
    jit_->set_executing(pc);
    fn(cpu_, mem_);
    if (chain_budget_) {
        uint32_t total = cpu_->packets;
        while (total < chain_budget_ && cpu_->pc == pc && !cpu_->halted && !cec_idle_pending() &&
               !cec_deliverable(cpu_) && mem_->allow_chain()) {
            const uint64_t c = ((uint64_t)cpu_->cycles[1] << 32 | cpu_->cycles[0]) + cpu_->packets;
            cpu_->cycles[0] = (uint32_t)c;
            cpu_->cycles[1] = cpu_->cycles[2] = (uint32_t)(c >> 32);
            cpu_->did_jump = false;
            cpu_->packets = 0;
            fn(cpu_, mem_);
            total += cpu_->packets ? cpu_->packets : 1;
        }
        cpu_->packets = total;
    }
    jit_->clear_executing();
#if BCORE_ENABLE_PROFILE
    profile_->stats.blocks_executed++;
#endif
    return true;
}

void Core::set_object_cache_dir(const std::string& dir) {
    object_cache_dir_ = dir;
    if (!dir.empty() && !pack_ && jit_) {
        // Everything besides the guest bytes that decides a block's code: lifter source, JIT settings,
        // and the memory model the lifter bakes in.
        translation_salt_ = std::string("bytes-v1|") + BCORE_SOURCE_FINGERPRINT + "|" + jit_->cache_salt() + "|fb=" +
                            std::to_string(mem_->fast_base()) + "|rl=" + std::to_string(mem_->rawmem_limit()) +
                            "|fm=" + std::to_string(mem_->is_fastmem()) + "|steps=" + std::to_string(cpu_->steps_remaining);
        Memory::FastRange ranges[8];
        const int nr = mem_->fast_ranges(ranges, 8);
        for (int i = 0; i < nr; ++i)
            translation_salt_ += "|fr=" + std::to_string(ranges[i].lo) + ":" + std::to_string(ranges[i].size) + "@" +
                                 std::to_string(ranges[i].host);
        pack_ = std::make_shared<ObjectPack>();
        pack_->SetSalt(translation_salt_);
        pack_->Open(dir);
    }
    if (jit_) {
        jit_->set_object_cache_dir(dir);
        jit_->set_object_pack(pack_.get());
    }
}

uint64_t Core::object_cache_hits() const { return cache_hits_base_ + (jit_ ? jit_->cache_hits() : 0); }
uint64_t Core::object_cache_misses() const { return cache_misses_base_ + (jit_ ? jit_->cache_misses() : 0); }

void Core::prefetch_pack() {
    prefetch_pcs_.clear();
    prefetch_pos_ = 0;
    if (!async_ || !jit_ || !jit_->worker_running()) return;
    ObjectPack* pack = jit_->object_pack();
    if (!pack || !pack->Enabled()) return;
    prefetch_pcs_ = pack->BlockPcs();
}

size_t Core::prefetch_step(size_t n) {
    size_t queued = 0;
    if (!async_ || !jit_ || !jit_->worker_running()) { prefetch_pos_ = prefetch_pcs_.size(); return 0; }
    for (const size_t end = std::min(prefetch_pcs_.size(), prefetch_pos_ + n); prefetch_pos_ < end; ++prefetch_pos_) {
        const uint32_t pc = prefetch_pcs_[prefetch_pos_];
        if (jit_->lookup(pc) || jit_->is_pending(pc)) continue;
        JitEngine::CompileJob job;
        job.pc = pc;
        if (!jit_->match_by_bytes(pc, mem_, job)) continue;
        job.bytes_hash = hash_code(pc, job.end_pc);
        jit_->enqueue_low(std::move(job));
        ++queued;
    }
    return queued;
}

void Core::invalidate() {
    if (jit_) { cache_hits_base_ += jit_->cache_hits(); cache_misses_base_ += jit_->cache_misses(); }
    jit_ = std::make_shared<JitEngine>();
    jit_->set_object_cache_dir(object_cache_dir_);
    jit_->set_object_pack(pack_.get());
    // Init cannot fail here unless it failed in Core::init already (same host,
    // same levels); if it ever does, stop rather than dereference a null LLJIT.
    if (!jit_->init(opt_level_, codegen_level_))
        llvm::report_fatal_error("bcore: JIT re-initialization failed in invalidate()");
    applyModuleTarget();
    pending_hits_.clear();
    if (async_) start_async_worker();
#if BCORE_ENABLE_PROFILE
    // The old engine's objects are gone; drop their map entries before the
    // host can observe stale addresses, then re-arm the listeners (including
    // jitdump) on the new engine: one registered on the old engine silently
    // misses everything compiled after this point.
    profile_->clearBlocks();
    attachProfiler();
#endif
}

std::tuple<std::string, uint32_t> Core::disassemble(uint32_t pc) {
    DisasmVisitor visitor;
    visitor.mem = mem_;

    std::ostringstream oss;
    visitor.set_out(&oss);

    int insn_len = decodeInstruction(visitor, pc);

    std::string text = oss.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
        text.pop_back();

    uint32_t next_pc = pc + static_cast<uint32_t>(insn_len);
    return {text, next_pc};
}

void Core::set_dump_ir(bool enable) {
    dump_ir_ = enable;
}


// ---------------------------------------------------------------------------
// Observability API
// ---------------------------------------------------------------------------

void Core::forEachCompiledBlock(
    const std::function<void(const BcoreBlockInfo&)>& fn) const {
#if BCORE_ENABLE_PROFILE
    if (profile_) {
        std::lock_guard<std::mutex> lk(profile_->blocks_mutex);
        for (const auto& [addr, b] : profile_->blocks) {
            fn(BcoreBlockInfo{reinterpret_cast<const void*>(addr), b.size,
                              b.guest_pc, b.variant, b.loaded_ns, 0});
        }
        if (profile_->listener_ok) return;
    }
#endif
    // Fallback (profiling disabled at build time or listener unavailable):
    // lookup-derived addresses, size unknown.
    if (jit_)
        jit_->forEachCompiled([&fn](const void* addr, uint32_t pc) {
            fn(BcoreBlockInfo{addr, /*host_size=*/0, pc, /*variant=*/0});
        });
}

void Core::setEventSink(BcoreEventSink* sink) {
#if BCORE_ENABLE_PROFILE
    profile_->sink = sink;
#else
    (void)sink;
#endif
}

BcoreStats Core::stats() const {
#if BCORE_ENABLE_PROFILE
    if (profile_)
        return profile_->stats;  // plain reads; dispatch is single-threaded
#endif
    return {};
}

bool Core::profileHasBlockSizes() const {
#if BCORE_ENABLE_PROFILE
    return profile_ && profile_->listener_ok;
#else
    return false;
#endif
}

void Core::set_perf_jitdump(bool enable) {
#if BCORE_ENABLE_PROFILE
    profile_->want_perf_jitdump = enable;
    if (!jit_) return;
    if (enable)
        bcore_profile::attach_perf_listener(*profile_, *jit_);
    else
        bcore_profile::detach_perf_listener(*profile_, *jit_);
#else
    (void)enable;
#endif
}

void Core::setProfileHistory(size_t capacity) {
#if BCORE_ENABLE_PROFILE
    std::lock_guard<std::mutex> lk(profile_->blocks_mutex);
    profile_->history_capacity = capacity;
    profile_->retired.clear();
    profile_->retired.reserve(capacity);
    profile_->history_dropped = 0;
    const auto now = capacity ? bcore_profile::monotonic_ns() : 0;
    for (auto& [addr, b] : profile_->blocks) b.loaded_ns = now;
#else
    (void)capacity;
#endif
}

void Core::forEachProfileBlock(const std::function<void(const BcoreBlockInfo&)>& fn) const {
#if BCORE_ENABLE_PROFILE
    std::lock_guard<std::mutex> lk(profile_->blocks_mutex);
    for (const auto& b : profile_->retired) fn(b);
    for (const auto& [addr, b] : profile_->blocks)
        fn({reinterpret_cast<const void*>(addr), b.size, b.guest_pc, b.variant, b.loaded_ns, 0});
    if (profile_->listener_ok) return;
#endif
    // No load listener (or profiling compiled out): live code from the
    // lookup table, sizes and lifetimes unknown, as in forEachCompiledBlock.
    if (jit_)
        jit_->forEachCompiled([&fn](const void* addr, uint32_t pc) {
            fn(BcoreBlockInfo{addr, /*host_size=*/0, pc, /*variant=*/0});
        });
}

uint64_t Core::profileHistoryDropped() const {
#if BCORE_ENABLE_PROFILE
    std::lock_guard<std::mutex> lk(profile_->blocks_mutex);
    return profile_->history_dropped;
#else
    return 0;
#endif
}
