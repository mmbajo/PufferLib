#ifndef PUFFER_NATIVE_CHECKPOINT_IO_H
#define PUFFER_NATIVE_CHECKPOINT_IO_H

// Small, dependency-free checkpoint container. Lengths are little endian and
// bounded by the caller's expected layout before allocating.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace puf_checkpoint {
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("training checkpoint: " + message);
}
class File {
    FILE* file_;
    bool writing_;
    std::string path_;
public:
    File(const std::string& path, bool writing) : writing_(writing), path_(path) {
        file_=fopen(path.c_str(),writing ? "wbx" : "rb");
        require(file_!=nullptr,"cannot open "+path);
    }
    ~File() { if(file_) fclose(file_); }
    File(const File&)=delete;
    File& operator=(const File&)=delete;
    void bytes(void* data, size_t size) {
        size_t done=writing_ ? fwrite(data,1,size,file_) : fread(data,1,size,file_);
        require(done==size,"short IO in "+path_);
    }
    uint64_t number(uint64_t value=0) {
        unsigned char data[8];
        for(int i=0;i<8;++i) data[i]=value>>(8*i);
        bytes(data,8); value=0;
        for(int i=0;i<8;++i) value|=uint64_t(data[i])<<(8*i);
        return value;
    }
    void expect(uint64_t value) { require(number(value)==value,"incompatible integer in "+path_); }
    std::string string(const std::string& value, size_t max_bytes=1u<<20) {
        uint64_t length=number(value.size());
        require(length<=max_bytes,"oversized string in "+path_);
        std::string result=writing_ ? value : std::string(size_t(length),'\0');
        bytes(result.data(),result.size()); return result;
    }
    void expect(const std::string& value) {
        require(string(value,value.size())==value,"incompatible metadata in "+path_);
    }
    void record(const std::string& kind) {
        const auto marker = string(kind + "-2", kind.size() + 2);
        require(marker != kind + "-1",
            "record version 1 requires feat/typed-supervision at 148add30; "
            "use its producing executable");
        require(marker == kind + "-2", "unsupported record version in " + path_);
    }
    void finish() {
        if (writing_) require(fflush(file_)==0 && fsync(fileno(file_))==0,"cannot sync "+path_);
        else require(fgetc(file_)==EOF && !ferror(file_),"trailing data in "+path_);
        int status=fclose(file_); file_=nullptr;
        require(status==0,"cannot close "+path_);
    }
};
} // namespace puf_checkpoint
#endif
