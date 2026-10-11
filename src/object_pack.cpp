#include <algorithm>
#include "object_pack.h"

#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/xxhash.h>

#include "mem.h"

namespace {

constexpr uint32_t kMagic = 0x4b504f42;  // "BOPK"

uint64_t Checksum(char type, const std::string& key, const char* data, size_t size) {
    std::string buf;
    buf.reserve(1 + key.size() + size);
    buf.push_back(type);
    buf += key;
    buf.append(data, size);
    return llvm::xxHash64(buf);
}

template <typename T> bool Get(const std::string& f, size_t& off, T& v) {
    if (off + sizeof(T) > f.size()) return false;
    std::memcpy(&v, f.data() + off, sizeof(T));
    off += sizeof(T);
    return true;
}

}  // namespace

void ObjectPack::Open(const std::string& dir) {
    if (fd_ >= 0 || dir.empty()) return;
    llvm::sys::fs::create_directories(dir);
    const std::string path = dir + "/objects.pack";
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) return;
    std::string file;
    char buf[1 << 16];
    ssize_t n;
    if (::lseek(fd_, 0, SEEK_SET) == 0)
        while ((n = ::read(fd_, buf, sizeof(buf))) > 0) file.append(buf, (size_t)n);
    Parse(file);
}

// Record: u32 magic, u8 type, u32 keyLen, key, u32 dataLen, data, u64 checksum.
void ObjectPack::Parse(const std::string& f) {
    size_t off = 0;
    while (off < f.size()) {
        uint32_t magic, klen, dlen;
        uint8_t type;
        uint64_t sum;
        if (!Get(f, off, magic) || magic != kMagic || !Get(f, off, type) || !Get(f, off, klen)) break;
        if (off + klen > f.size()) break;
        std::string key = f.substr(off, klen);
        off += klen;
        if (!Get(f, off, dlen) || off + dlen > f.size()) break;
        const char* data = f.data() + off;
        off += dlen;
        if (!Get(f, off, sum) || sum != Checksum((char)type, key, data, dlen)) break;
        if (type == 'O') {
            objects_.emplace(std::move(key), std::string(data, dlen));
        } else if (type == 'B' && dlen >= 20) {
            uint32_t pc;
            BlockBytes b;
            std::memcpy(&pc, data, 4);
            std::memcpy(&b.len, data + 4, 4);
            std::memcpy(&b.fallthrough, data + 8, 4);
            std::memcpy(&b.hash, data + 12, 8);
            b.objKey.assign(data + 20, dlen - 20);
            if (key == salt_) blocks_[pc].push_back(std::move(b));
        }
    }
}

void ObjectPack::Append(char type, const std::string& key, const char* data, size_t size) {
    if (fd_ < 0) return;
    std::string rec;
    const uint32_t magic = kMagic, klen = (uint32_t)key.size(), dlen = (uint32_t)size;
    const uint8_t t = (uint8_t)type;
    const uint64_t sum = Checksum(type, key, data, size);
    rec.append((const char*)&magic, 4);
    rec.append((const char*)&t, 1);
    rec.append((const char*)&klen, 4);
    rec += key;
    rec.append((const char*)&dlen, 4);
    rec.append(data, size);
    rec.append((const char*)&sum, 8);
    ssize_t w = ::write(fd_, rec.data(), rec.size());
    (void)w;
}

const std::string* ObjectPack::FindObject(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = objects_.find(key);
    return it == objects_.end() ? nullptr : &it->second;
}

void ObjectPack::AddObject(const std::string& key, const char* data, size_t size) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0 || objects_.count(key)) return;
    objects_.emplace(key, std::string(data, size));
    Append('O', key, data, size);
}

const std::vector<ObjectPack::BlockBytes>* ObjectPack::FindBlocks(uint32_t pc) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = blocks_.find(pc);
    return it == blocks_.end() ? nullptr : &it->second;
}

std::vector<uint32_t> ObjectPack::BlockPcs() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<uint32_t> pcs;
    pcs.reserve(blocks_.size());
    for (const auto& kv : blocks_) pcs.push_back(kv.first);
    std::sort(pcs.begin(), pcs.end());
    return pcs;
}

void ObjectPack::AddBlock(uint32_t pc, const BlockBytes& b) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0) return;
    for (const auto& e : blocks_[pc])
        if (e.len == b.len && e.hash == b.hash && e.objKey == b.objKey) return;
    blocks_[pc].push_back(b);
    std::string data(20, '\0');
    std::memcpy(&data[0], &pc, 4);
    std::memcpy(&data[4], &b.len, 4);
    std::memcpy(&data[8], &b.fallthrough, 4);
    std::memcpy(&data[12], &b.hash, 8);
    data += b.objKey;
    // The salt is the record key: index entries only apply to identical translation settings.
    Append('B', salt_, data.data(), data.size());
}

uint64_t ObjectPack::HashBytes(Memory* mem, uint32_t pc, uint32_t len) const {
    std::string buf = salt_;
    buf.append((const char*)&pc, 4);
    buf.reserve(buf.size() + len);
    for (uint32_t i = 0; i < len; ++i) buf.push_back((char)mem->read8(pc + i));
    return llvm::xxHash64(buf);
}
