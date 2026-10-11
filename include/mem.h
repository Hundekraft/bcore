#pragma once

#include <cstdint>
#include <cstring>
#include <string>

class Memory {
public:
    virtual ~Memory() = default;

    virtual uint32_t base() const = 0;
    virtual uint32_t size() const = 0;

    virtual uint8_t  read8(uint32_t addr) const = 0;
    virtual uint16_t read16(uint32_t addr) const = 0;
    virtual uint32_t read32(uint32_t addr) const = 0;

    virtual void write8(uint32_t addr, uint8_t val) = 0;
    virtual void write16(uint32_t addr, uint16_t val) = 0;
    virtual void write32(uint32_t addr, uint32_t val) = 0;

    virtual const uint8_t* raw() const = 0;

    // Host address corresponding to emulator address 0x0.
    // In fastmem mode: (uintptr_t)mmap_base
    // In normal mode: (uintptr_t)data_.data() - base_
    virtual uintptr_t fast_base() const = 0;

    // Returns true if memory uses direct host-address (mmap) mode.
    // When true, JIT emits direct loads/stores using fast_base().
    virtual bool is_fastmem() const { return false; }

    // Upper bound for the inline (non-fastmem) fast-path: addresses below this
    // value can be accessed directly via fast_base() + addr without an extern call.
    // Returns 0 in fastmem mode (not needed there).
    virtual uint32_t rawmem_limit() const { return is_fastmem() ? 0u : (base() + size()); }

    // Core::run may re-run a block that jumps back to its own start without returning to the host
    // (see Core::set_chain_budget). A host returns false once something it must react to between blocks
    // happened during the block, e.g. an MMR access.
    virtual bool allow_chain() const { return true; }

    // More plain memory the JIT may access directly besides [0, rawmem_limit()): guest [lo, lo + size) at
    // host + (addr - lo). The host pointers must stay the same across runs (they are baked into cached code).
    struct FastRange { uint32_t lo, size; uintptr_t host; };
    virtual int fast_ranges(FastRange* out, int max) const { (void)out; (void)max; return 0; }
};

// Appends guest bytes [addr, addr + len) to out: copied straight from host memory when the whole range is
// plain memory ([0, rawmem_limit()) or one fast range), read8 by read8 otherwise. Hashing guest code
// (pack matching, stale-compile checks) goes through here.
inline void append_guest_bytes(const Memory* mem, uint32_t addr, uint32_t len, std::string& out) {
    const uint64_t end = (uint64_t)addr + len;
    const uint8_t* src = nullptr;
    if (!mem->is_fastmem() && end <= mem->rawmem_limit()) {
        src = (const uint8_t*)(mem->fast_base() + addr);
    } else {
        Memory::FastRange r[8];
        const int n = mem->fast_ranges(r, 8);
        for (int i = 0; i < n && !src; ++i)
            if (addr >= r[i].lo && end <= (uint64_t)r[i].lo + r[i].size) src = (const uint8_t*)(r[i].host + (addr - r[i].lo));
    }
    if (src) { out.append((const char*)src, len); return; }
    out.reserve(out.size() + len);
    for (uint32_t i = 0; i < len; ++i) out.push_back((char)mem->read8(addr + i));
}

// extern "C" wrappers for JIT-generated code
extern "C" {
    uint8_t  mem_read8(Memory* mem, uint32_t addr);
    uint16_t mem_read16(Memory* mem, uint32_t addr);
    uint32_t mem_read32(Memory* mem, uint32_t addr);
    void     mem_write8(Memory* mem, uint32_t addr, uint8_t val);
    void     mem_write16(Memory* mem, uint32_t addr, uint16_t val);
    void     mem_write32(Memory* mem, uint32_t addr, uint32_t val);
}
