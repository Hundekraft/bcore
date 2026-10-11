#include <cstdlib>
#include "jit_engine.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "object_pack.h"
#include "syscall_emu.h"
#include "cec.h"
#include "mmr.h"
#include "cpu_state.h"

// Object files on disk, one per block, named <key>.o. Only modules tagged by tag_cached_module (identifier
// "bcache-<key>") take part; anything else compiles normally and is not stored.
class DiskObjectCache : public llvm::ObjectCache {
public:
    std::string dir;
    ObjectPack* pack = nullptr;

    static bool keyOf(const llvm::Module* m, std::string& key) {
        const std::string& id = m->getModuleIdentifier();
        if (id.rfind("bcache-", 0) != 0) return false;
        key = id.substr(7);
        return true;
    }
    std::string path(const std::string& key) const { return dir + "/" + key + ".o"; }

    void notifyObjectCompiled(const llvm::Module* m, llvm::MemoryBufferRef obj) override {
        std::string key;
        if (dir.empty() || !keyOf(m, key)) return;
        if (pack && pack->Enabled()) {
            pack->AddObject(key, obj.getBufferStart(), obj.getBufferSize());
            return;
        }
        // Write to a temp file and rename, so a concurrent reader never sees a partial object.
        const std::string tmp = path(key) + ".tmp" + std::to_string(::getpid());
        std::error_code ec;
        {
            llvm::raw_fd_ostream os(tmp, ec, llvm::sys::fs::OF_None);
            if (ec) return;
            os << obj.getBuffer();
        }
        if (llvm::sys::fs::rename(tmp, path(key))) llvm::sys::fs::remove(tmp);
    }
    std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module* m) override {
        std::string key;
        if (dir.empty() || !keyOf(m, key)) return nullptr;
        if (pack)
            if (const std::string* o = pack->FindObject(key))
                return llvm::MemoryBuffer::getMemBufferCopy(*o);
        auto buf = llvm::MemoryBuffer::getFile(path(key), /*IsText=*/false, /*RequiresNullTerminator=*/false);
        if (!buf) return nullptr;
        // Older caches kept one file per object: move hits into the pack so the byte index can use them.
        if (pack && pack->Enabled()) pack->AddObject(key, (*buf)->getBufferStart(), (*buf)->getBufferSize());
        return std::move(*buf);
    }
};

JitEngine::JitEngine() : objcache_(std::make_unique<DiskObjectCache>()) {}
JitEngine::~JitEngine() {
    if (!workers_.empty()) {
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            stop_ = true;
        }
        queue_cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
}

void JitEngine::start_worker(Lifter lifter, unsigned count) {
    if (!workers_.empty()) return;
    lifter_ = std::move(lifter);
    for (unsigned i = 0; i < std::max(1u, count); ++i) workers_.emplace_back([this] { worker_loop(); });
}

void JitEngine::enqueue(CompileJob job) {
    pending_.insert(job.pc);
    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        queue_.push_back(std::move(job));   // in request order; hot blocks are moved up by prioritize()
    }
    queue_cv_.notify_one();
}

void JitEngine::prioritize(uint32_t pc) {
    std::lock_guard<std::mutex> lock(queue_mu_);
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->pc != pc) continue;
        if (it != queue_.begin()) {
            CompileJob job = std::move(*it);
            queue_.erase(it);
            queue_.push_front(std::move(job));
        }
        break;
    }
}

void JitEngine::wait_for(uint32_t pc) {
    prioritize(pc);
    std::unique_lock<std::mutex> lock(queue_mu_);
    done_cv_.wait(lock, [&] {
        for (const auto& d : done_)
            if (d.pc == pc) return true;
        return false;
    });
}

std::vector<JitEngine::CompiledBlock> JitEngine::take_compiled() {
    std::vector<CompiledBlock> out;
    if (!has_done_.load(std::memory_order_acquire)) return out;
    {
        std::lock_guard<std::mutex> lock(queue_mu_);
        out.swap(done_);
        has_done_.store(false, std::memory_order_relaxed);
    }
    for (const auto& d : out) {
        pending_.erase(d.pc);
        if (d.from_pack && !d.fn) pack_failed_.insert(d.pc);
    }
    return out;
}

void JitEngine::worker_loop() {
    for (;;) {
        CompileJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mu_);
            queue_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        CompiledBlock done;
        if (!job.obj_key.empty()) {   // byte-index hit: link the stored object, no lift or compile
            done.pc = job.pc;
            done.end_pc = job.end_pc;
            done.fallthrough_pc = job.fallthrough_pc;
            done.bytes_hash = job.bytes_hash;
            done.from_pack = true;
            done.fn = load_object(job.pc, job.obj_key);
            {
                std::lock_guard<std::mutex> lock(queue_mu_);
                done_.push_back(std::move(done));
                has_done_.store(true, std::memory_order_release);
            }
            done_cv_.notify_all();
            continue;
        }
        bool cached = false;
        if (!job.module) {
            // Queued by pc only: lift here, so the CPU thread never waits for the lifter either.
            if (!lifter_ || !lifter_(job.pc, job)) {
                done.pc = job.pc;
                std::lock_guard<std::mutex> lock(queue_mu_);
                done_.push_back(std::move(done));
                has_done_.store(true, std::memory_order_release);
                done_cv_.notify_all();
                continue;
            }
            cached = tag_cached_module(*job.module);
            job.module_id = job.module->getModuleIdentifier();
        }
        if (!cached) optimize_module(*job.module);
        done.pc = job.pc;
        done.end_pc = job.end_pc;
        done.fallthrough_pc = job.fallthrough_pc;
        done.bytes_hash = job.bytes_hash;
        done.module_id = job.module_id;
        done.fn = materialize(job.pc, std::move(job.module), std::move(job.context));
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            done_.push_back(std::move(done));
            has_done_.store(true, std::memory_order_release);
        }
        done_cv_.notify_all();
    }
}

// Helper to create a JITEvaluatedSymbol from a function pointer
static llvm::JITEvaluatedSymbol sym_from_ptr(void* ptr) {
    return llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(ptr),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
}

namespace {
// Very large blocks (long straight-line guest code lifts to 5k-20k IR instructions) spend 40-250 ms in the
// backend at CodeGenOpt::Default, and every compile stalls the guest. Those use the fast backend
// (CodeGenOpt::None: FastISel, fast register allocation); everything else keeps the configured level. The
// choice depends only on the optimized module, so it is reproducible and the object cache stays valid.
// BCORE_FAST_CG_IR=N sets the IR instruction threshold (default 1500, 0 = off).
class SizeTieredCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
public:
    // Compiles on whichever thread materializes (async compile runs several workers), so each thread gets its
    // own TargetMachines: they are not safe to share.
    SizeTieredCompiler(llvm::orc::JITTargetMachineBuilder jtmb, const llvm::TargetOptions& options,
                       llvm::ObjectCache* cache, unsigned threshold)
        : IRCompiler(llvm::orc::irManglingOptionsFromTargetOptions(options)),
          jtmb_(std::move(jtmb)), cache_(cache), threshold_(threshold) {}

    llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> operator()(llvm::Module& m) override {
        auto per = get();
        if (!per) return per.takeError();
        if (threshold_ && m.getInstructionCount() > threshold_) return (*(*per)->fast)(m);
        return (*(*per)->normal)(m);
    }

private:
    struct PerThread {
        std::unique_ptr<llvm::TargetMachine> normal_tm, fast_tm;
        std::unique_ptr<llvm::orc::SimpleCompiler> normal, fast;
    };
    llvm::Expected<PerThread*> get() {
        std::lock_guard<std::mutex> lock(mu_);
        auto& p = per_[std::this_thread::get_id()];
        if (!p) {
            auto n = jtmb_.createTargetMachine();
            if (!n) return n.takeError();
            auto fb = jtmb_;
            fb.setCodeGenOptLevel(llvm::CodeGenOpt::None);
            auto f = fb.createTargetMachine();
            if (!f) return f.takeError();
            auto pt = std::make_unique<PerThread>();
            pt->normal_tm = std::move(*n);
            pt->fast_tm = std::move(*f);
            pt->normal = std::make_unique<llvm::orc::SimpleCompiler>(*pt->normal_tm, cache_);
            pt->fast = std::make_unique<llvm::orc::SimpleCompiler>(*pt->fast_tm, cache_);
            p = std::move(pt);
        }
        return p.get();
    }
    llvm::orc::JITTargetMachineBuilder jtmb_;
    llvm::ObjectCache* cache_;
    unsigned threshold_;
    std::mutex mu_;
    std::unordered_map<std::thread::id, std::unique_ptr<PerThread>> per_;
};
}  // namespace

bool JitEngine::init(int opt_level, int codegen_level) {
    opt_level_ = opt_level;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // The IR pipeline and the backend are separate knobs: the backend level
    // decides instruction selection and register allocation, the IR pipeline
    // what the backend is given, and each has its own compile-time cost.
    const int cg = codegen_level < 0 ? opt_level : codegen_level;
    llvm::CodeGenOpt::Level cg_level =
        cg <= 0 ? llvm::CodeGenOpt::None :
        cg == 1 ? llvm::CodeGenOpt::Less :
        cg == 3 ? llvm::CodeGenOpt::Aggressive :
                  llvm::CodeGenOpt::Default;

    auto jtmb_or_err = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb_or_err) {
        llvm::errs() << "Failed to detect host: " << jtmb_or_err.takeError() << "\n";
        return false;
    }
    jtmb_or_err->setCodeGenOptLevel(cg_level);

    // The IR pipeline's TargetMachine comes from a copy of the compiler's
    // builder, so its TTI describes exactly the CPU and features the code will
    // be generated for. Without one, PassBuilder's TargetIRAnalysis is the
    // generic implementation: zero vector registers for LoopVectorize/SLP, no
    // x86 unrolling preferences, generic SimplifyCFG/LICM/JumpThreading costs.
    // Single-threaded use only, like all of bcore's compilation (X86's
    // subtarget cache in TargetMachine is unlocked).
    {
        auto machine = llvm::orc::JITTargetMachineBuilder(*jtmb_or_err).createTargetMachine();
        if (!machine) {
            llvm::errs() << "Failed to create pass TargetMachine: " << machine.takeError() << "\n";
            return false;
        }
        pass_machine_ = std::move(*machine);
    }
    // Clang's -O1 leaves both vectorizers off and -O2/-O3 turn both on; the
    // PipelineTuningOptions defaults (loop vectorization on, SLP off) match
    // neither, so follow clang.
    llvm::PipelineTuningOptions tuning;
    tuning.LoopVectorization = opt_level >= 2;
    tuning.LoopInterleaving = opt_level >= 2;
    tuning.SLPVectorization = opt_level >= 2;
    pb_ = std::make_unique<llvm::PassBuilder>(pass_machine_.get(), tuning);
    jtmb_ = std::make_unique<llvm::orc::JITTargetMachineBuilder>(*jtmb_or_err);
    owner_thread_ = std::this_thread::get_id();

    // Everything besides the bitcode that decides the generated code goes into the cache key.
    cache_salt_ = std::string("bcore-objcache-v1|") + LLVM_VERSION_STRING + "|" +
                  jtmb_or_err->getTargetTriple().str() + "|" + jtmb_or_err->getCPU() + "|" +
                  jtmb_or_err->getFeatures().getString() + "|O" + std::to_string(opt_level) + "|CG" +
                  std::to_string(cg);

    DiskObjectCache* objcache = objcache_.get();
    auto jit_or_err = llvm::orc::LLJITBuilder()
        .setJITTargetMachineBuilder(std::move(*jtmb_or_err))
        .setCompileFunctionCreator(
            [objcache](llvm::orc::JITTargetMachineBuilder jtmb)
                -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
                auto tm = jtmb.createTargetMachine();
                if (!tm) return tm.takeError();
                const char* env = getenv("BCORE_FAST_CG_IR");
                const unsigned threshold = env ? (unsigned)strtoul(env, nullptr, 10) : 1500;
                return std::make_unique<SizeTieredCompiler>(jtmb, (*tm)->Options, objcache, threshold);
            })
        .create();
    if (!jit_or_err) {
        llvm::errs() << "Failed to create LLJIT: " << jit_or_err.takeError() << "\n";
        return false;
    }
    jit_ = std::move(*jit_or_err);

    // Pre-allocate hash buckets to avoid rehash overhead during execution.
    cache_.reserve(1024);

    // Manually register all extern "C" symbols that JIT'd code may call.
    // This is more reliable than DynamicLibrarySearchGenerator which depends
    // on -rdynamic / ENABLE_EXPORTS.
    auto& jd = jit_->getMainJITDylib();

    llvm::orc::SymbolMap symbols;
    symbols[jit_->mangleAndIntern("mem_read8")]           = sym_from_ptr(reinterpret_cast<void*>(&mem_read8));
    symbols[jit_->mangleAndIntern("mem_read16")]          = sym_from_ptr(reinterpret_cast<void*>(&mem_read16));
    symbols[jit_->mangleAndIntern("mem_read32")]          = sym_from_ptr(reinterpret_cast<void*>(&mem_read32));
    symbols[jit_->mangleAndIntern("mem_write8")]          = sym_from_ptr(reinterpret_cast<void*>(&mem_write8));
    symbols[jit_->mangleAndIntern("mem_write16")]         = sym_from_ptr(reinterpret_cast<void*>(&mem_write16));
    symbols[jit_->mangleAndIntern("mem_write32")]         = sym_from_ptr(reinterpret_cast<void*>(&mem_write32));
    symbols[jit_->mangleAndIntern("bfin_syscall")]        = sym_from_ptr(reinterpret_cast<void*>(&bfin_syscall));
    symbols[jit_->mangleAndIntern("bfin_putchar")]        = sym_from_ptr(reinterpret_cast<void*>(&bfin_putchar));
    symbols[jit_->mangleAndIntern("cec_exception")]       = sym_from_ptr(reinterpret_cast<void*>(&cec_exception));
    symbols[jit_->mangleAndIntern("cec_raise")]           = sym_from_ptr(reinterpret_cast<void*>(&cec_raise));
    symbols[jit_->mangleAndIntern("cec_return_rti")]      = sym_from_ptr(reinterpret_cast<void*>(&cec_return_rti));
    symbols[jit_->mangleAndIntern("cec_return_rtx")]      = sym_from_ptr(reinterpret_cast<void*>(&cec_return_rtx));
    symbols[jit_->mangleAndIntern("cec_return_rtn")]      = sym_from_ptr(reinterpret_cast<void*>(&cec_return_rtn));
    symbols[jit_->mangleAndIntern("cec_return_rte")]      = sym_from_ptr(reinterpret_cast<void*>(&cec_return_rte));
    symbols[jit_->mangleAndIntern("cec_cli")]             = sym_from_ptr(reinterpret_cast<void*>(&cec_cli));
    symbols[jit_->mangleAndIntern("cec_sti")]             = sym_from_ptr(reinterpret_cast<void*>(&cec_sti));
    symbols[jit_->mangleAndIntern("cec_push_reti")]       = sym_from_ptr(reinterpret_cast<void*>(&cec_push_reti));
    symbols[jit_->mangleAndIntern("cec_pop_reti")]        = sym_from_ptr(reinterpret_cast<void*>(&cec_pop_reti));
    symbols[jit_->mangleAndIntern("cec_check_sup")]       = sym_from_ptr(reinterpret_cast<void*>(&cec_check_sup));
    symbols[jit_->mangleAndIntern("cec_is_user_mode")]    = sym_from_ptr(reinterpret_cast<void*>(&cec_is_user_mode));
    symbols[jit_->mangleAndIntern("cec_idle")]            = sym_from_ptr(reinterpret_cast<void*>(&cec_idle));
    symbols[jit_->mangleAndIntern("cec_check_pending")]   = sym_from_ptr(reinterpret_cast<void*>(&cec_check_pending));
    symbols[jit_->mangleAndIntern("bfin_hwloop_step")]    = sym_from_ptr(reinterpret_cast<void*>(&bfin_hwloop_step));
    // Not called by the lifter. With an IR pipeline, LoopIdiomRecognize can turn
    // a guest fill/copy loop into llvm.memset/memcpy/memmove, and the backend
    // lowers a non-constant length to the libc call; resolve it here like every
    // other symbol rather than through a process-wide search generator.
    symbols[jit_->mangleAndIntern("memset")]              = sym_from_ptr(reinterpret_cast<void*>(&::memset));
    symbols[jit_->mangleAndIntern("memcpy")]              = sym_from_ptr(reinterpret_cast<void*>(&::memcpy));
    symbols[jit_->mangleAndIntern("memmove")]             = sym_from_ptr(reinterpret_cast<void*>(&::memmove));

    if (auto err = jd.define(llvm::orc::absoluteSymbols(symbols))) {
        llvm::errs() << "Failed to register symbols: " << err << "\n";
        return false;
    }

    return true;
}

void JitEngine::optimize_module(llvm::Module& mod) {
    if (opt_level_ == 0) return;
    // Compile workers each get their own pass TargetMachine and PassBuilder (neither is thread-safe).
    llvm::TargetMachine* machine = pass_machine_.get();
    llvm::PassBuilder* pb = pb_.get();
    if (std::this_thread::get_id() != owner_thread_) {
        std::lock_guard<std::mutex> lock(pass_mu_);
        auto& wp = worker_passes_[std::this_thread::get_id()];
        if (!wp.machine) {
            auto m = llvm::orc::JITTargetMachineBuilder(*jtmb_).createTargetMachine();
            if (!m) { llvm::consumeError(m.takeError()); return; }
            wp.machine = std::move(*m);
            llvm::PipelineTuningOptions tuning;
            tuning.LoopVectorization = opt_level_ >= 2;
            tuning.LoopInterleaving = opt_level_ >= 2;
            tuning.SLPVectorization = opt_level_ >= 2;
            wp.pb = std::make_unique<llvm::PassBuilder>(wp.machine.get(), tuning);
        }
        machine = wp.machine.get();
        pb = wp.pb.get();
    }

    llvm::OptimizationLevel lvl =
        opt_level_ == 1 ? llvm::OptimizationLevel::O1 :
        opt_level_ == 3 ? llvm::OptimizationLevel::O3 :
                          llvm::OptimizationLevel::O2;

    llvm::LoopAnalysisManager     lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager    cgam;
    llvm::ModuleAnalysisManager   mam;

    // PassBuilder(TM) registers the same TargetIRAnalysis itself; registering
    // it first makes that explicit (the later default registration is then a
    // no-op).
    fam.registerPass([&] { return machine->getTargetIRAnalysis(); });
    pb->registerModuleAnalyses(mam);
    pb->registerCGSCCAnalyses(cgam);
    pb->registerFunctionAnalyses(fam);
    pb->registerLoopAnalyses(lam);
    pb->crossRegisterProxies(lam, fam, cgam, mam);

    // Rebuild per-module pipeline each call (analysis managers are fresh).
    // passBuilder itself is reused across calls (avoids plugin registration overhead).
    llvm::ModulePassManager mpm = pb->buildPerModuleDefaultPipeline(lvl);
    mpm.run(mod, mam);
}

BbFunc JitEngine::materialize(uint32_t pc, std::unique_ptr<llvm::Module> module,
                              std::unique_ptr<llvm::LLVMContext> ctx) {
    auto tsm = llvm::orc::ThreadSafeModule(std::move(module), std::move(ctx));

    if (auto err = jit_->addIRModule(std::move(tsm))) {
        llvm::errs() << "Failed to add module for PC=0x" << llvm::format_hex(pc, 8)
                     << ": " << err << "\n";
        return nullptr;
    }

    // Look up the function
    char name[32];
    snprintf(name, sizeof(name), "bb_0x%08x", pc);

    auto sym = jit_->lookup(name);
    if (!sym) {
        llvm::errs() << "Failed to lookup " << name << ": " << sym.takeError() << "\n";
        return nullptr;
    }
    return reinterpret_cast<BbFunc>(sym->getValue());
}

bool JitEngine::addModule(uint32_t pc, std::unique_ptr<llvm::Module> module,
                          std::unique_ptr<llvm::LLVMContext> ctx,
                          uint32_t fallthrough_pc) {
    BbFunc fn = materialize(pc, std::move(module), std::move(ctx));
    if (!fn) return false;
    cache_[pc] = BBInfo{fn, fallthrough_pc};
    return true;
}

void JitEngine::set_object_cache_dir(const std::string& dir) {
    objcache_->dir = dir;
    if (!dir.empty()) llvm::sys::fs::create_directories(dir);
}

bool JitEngine::tag_cached_module(llvm::Module& mod) {
    if (objcache_->dir.empty()) return false;
    llvm::SmallVector<char, 0> bc;
    {
        llvm::raw_svector_ostream os(bc);
        llvm::WriteBitcodeToFile(mod, os);
    }
    llvm::SHA256 sha;
    sha.update(cache_salt_);
    sha.update(llvm::StringRef(bc.data(), bc.size()));
    const auto digest = sha.final();
    static const char* hex = "0123456789abcdef";
    std::string key;
    for (uint8_t b : digest) { key += hex[b >> 4]; key += hex[b & 15]; }
    mod.setModuleIdentifier("bcache-" + key);
    const bool hit = (objcache_->pack && objcache_->pack->FindObject(key)) || llvm::sys::fs::exists(objcache_->path(key));
    (hit ? cache_hits_ : cache_misses_)++;
    return hit;
}

void JitEngine::set_object_pack(ObjectPack* pack) { objcache_->pack = pack; }

BbFunc JitEngine::load_object(uint32_t pc, const std::string& key) {
    ObjectPack* pack = objcache_->pack;
    const std::string* obj = pack ? pack->FindObject(key) : nullptr;
    if (!obj) return nullptr;
    if (auto err = jit_->addObjectFile(llvm::MemoryBuffer::getMemBufferCopy(*obj))) {
        llvm::consumeError(std::move(err));
        return nullptr;
    }
    char name[32];
    snprintf(name, sizeof(name), "bb_0x%08x", pc);
    auto sym = jit_->lookup(name);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return nullptr;
    }
    ++cache_hits_;
    return reinterpret_cast<BbFunc>(sym->getValue());
}

bool JitEngine::match_by_bytes(uint32_t pc, Memory* mem, CompileJob& job) {
    ObjectPack* pack = objcache_->pack;
    if (!pack || !pack->Enabled() || pack_failed_.count(pc)) return false;
    const auto* cands = pack->FindBlocks(pc);
    if (!cands) return false;
    for (const auto& b : *cands) {
        if (pack->HashBytes(mem, pc, b.len) != b.hash || !pack->FindObject(b.objKey)) continue;
        job.obj_key = b.objKey;
        job.end_pc = pc + b.len;
        job.fallthrough_pc = b.fallthrough;
        return true;
    }
    return false;
}

bool JitEngine::try_load_by_bytes(uint32_t pc, Memory* mem) {
    ObjectPack* pack = objcache_->pack;
    if (!pack || !pack->Enabled()) return false;
    const auto* cands = pack->FindBlocks(pc);
    if (!cands) return false;
    for (const auto& b : *cands) {
        if (pack->HashBytes(mem, pc, b.len) != b.hash) continue;
        if (!pack->FindObject(b.objKey)) continue;
        BbFunc fn = load_object(pc, b.objKey);
        if (!fn) return false;
        cache_[pc] = BBInfo{fn, b.fallthrough};
        return true;
    }
    return false;
}

void JitEngine::record_block_bytes(uint32_t pc, uint32_t end_pc, uint32_t fallthrough_pc, Memory* mem,
                                   const std::string& module_id) {
    ObjectPack* pack = objcache_->pack;
    if (!pack || !pack->Enabled() || module_id.rfind("bcache-", 0) != 0 || end_pc <= pc) return;
    ObjectPack::BlockBytes b;
    b.len = end_pc - pc;
    b.fallthrough = fallthrough_pc;
    b.hash = pack->HashBytes(mem, pc, b.len);
    b.objKey = module_id.substr(7);
    if (pack->FindObject(b.objKey)) pack->AddBlock(pc, b);
}

BbFunc JitEngine::lookup(uint32_t pc) {
    auto it = cache_.find(pc);
    if (it != cache_.end())
        return it->second.fn;
    return nullptr;
}

bool JitEngine::registerJITEventListener(llvm::JITEventListener& L) {
    // Runtime check, not an assumption: LLJIT's default on x86-64 Linux is
    // RTDyldObjectLinkingLayer (LLJIT.h), but a custom linking-layer creator
    // or another platform may differ.
    auto* rtdyld = llvm::dyn_cast<llvm::orc::RTDyldObjectLinkingLayer>(
        &jit_->getObjLinkingLayer());
    if (!rtdyld)
        return false;
    rtdyld->registerJITEventListener(L);
    return true;
}

void JitEngine::unregisterJITEventListener(llvm::JITEventListener& L) {
    if (auto* rtdyld = llvm::dyn_cast<llvm::orc::RTDyldObjectLinkingLayer>(&jit_->getObjLinkingLayer()))
        rtdyld->unregisterJITEventListener(L);
}
