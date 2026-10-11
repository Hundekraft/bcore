#pragma once

// Persistent JIT cache in one append-only file (objects.pack) instead of one file per block.
//
// Records: 'O' = compiled object for a module key (SHA-256 of the block's unoptimized bitcode + settings),
//          'B' = byte index: the guest code bytes a block was lifted from (pc, length, hash) -> object key,
//                so a later run can load the object without lifting the block again.
// Every record carries an xxHash64 checksum; loading stops at the first damaged record. Appends are one
// write() each with O_APPEND, so concurrent emulator processes only ever add whole records.

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class Memory;

class ObjectPack {
public:
    struct BlockBytes {
        uint32_t len = 0;           // guest bytes [pc, pc + len) the block was lifted from
        uint32_t fallthrough = 0;
        uint64_t hash = 0;          // HashBytes() of those bytes
        std::string objKey;
    };

    // Opens (and loads) dir/objects.pack; empty dir = disabled.
    void Open(const std::string& dir);
    bool Enabled() const { return fd_ >= 0; }

    const std::string* FindObject(const std::string& key) const;
    void AddObject(const std::string& key, const char* data, size_t size);

    // Candidates for pc whose bytes still match guest memory (checked by the caller with HashBytes).
    const std::vector<BlockBytes>* FindBlocks(uint32_t pc) const;
    // Every pc with an entry in the byte index (for preloading).
    std::vector<uint32_t> BlockPcs() const;
    void AddBlock(uint32_t pc, const BlockBytes& b);

    // Hash of the guest code bytes [pc, pc + len) plus pc and the translation settings salt.
    uint64_t HashBytes(Memory* mem, uint32_t pc, uint32_t len) const;
    void SetSalt(const std::string& salt) { salt_ = salt; }

    uint64_t ObjectsLoaded() const { return objects_.size(); }

private:
    void Append(char type, const std::string& key, const char* data, size_t size);
    void Parse(const std::string& file);

    int fd_ = -1;
    std::string salt_;
    // Objects are added by the async compile worker (notifyObjectCompiled) while the CPU thread looks them up.
    // Elements of objects_ are node-stable, so FindObject's pointer stays valid after the lock is released.
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::string> objects_;
    std::unordered_map<uint32_t, std::vector<BlockBytes>> blocks_;
};
