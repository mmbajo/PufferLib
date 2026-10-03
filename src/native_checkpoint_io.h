#ifndef PUFFER_NATIVE_CHECKPOINT_IO_H
#define PUFFER_NATIVE_CHECKPOINT_IO_H

// Small, dependency-free checkpoint container. Lengths are little endian and
// bounded by the caller's expected layout before allocating. SHA-256 detects
// accidental corruption; checkpoints are trusted local artifacts, not signed.
#include <array>
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
using Digest = std::array<unsigned char, 32>;
class SHA256 {
    uint32_t h_[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    unsigned char buffer_[64]{};
    uint64_t count_ = 0;
    static uint32_t rotate(uint32_t x, int n) { return (x >> n) | (x << (32-n)); }
    void block(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i=0;i<16;++i) w[i]=(uint32_t(p[4*i])<<24)|(uint32_t(p[4*i+1])<<16)|
            (uint32_t(p[4*i+2])<<8)|p[4*i+3];
        for (int i=16;i<64;++i) {
            uint32_t a=w[i-15], b=w[i-2];
            w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
        }
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
        for (int i=0;i<64;++i) {
            uint32_t t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=h;
    }
public:
    void update(const void* data, size_t size) {
        const auto* p=static_cast<const unsigned char*>(data);
        while (size) {
            size_t at=count_%64, n=64-at; if (n>size) n=size;
            memcpy(buffer_+at,p,n); count_+=n; p+=n; size-=n;
            if (count_%64==0) block(buffer_);
        }
    }
    Digest finish() const {
        SHA256 copy=*this;
        uint64_t bits=count_*8;
        unsigned char tail[128]{}; tail[0]=0x80;
        size_t padding=(count_%64<56 ? 56 : 120)-count_%64;
        for(int i=0;i<8;++i) tail[padding+i]=static_cast<unsigned char>(bits>>(56-8*i));
        copy.update(tail,padding+8);
        Digest digest{};
        for(int i=0;i<32;++i) digest[i]=copy.h_[i/4]>>(24-8*(i%4));
        return digest;
    }
};
inline std::string hex(const Digest& digest) {
    const char* alphabet="0123456789abcdef";
    std::string out;
    for (unsigned char c : digest) { out+=alphabet[c>>4]; out+=alphabet[c&15]; }
    return out;
}
inline Digest hash_file(const std::string& path) {
    FILE* f=fopen(path.c_str(),"rb");
    require(f!=nullptr,"cannot open "+path);
    SHA256 hash; unsigned char buffer[65536]; size_t n;
    while ((n=fread(buffer,1,sizeof(buffer),f))) hash.update(buffer,n);
    bool ok=!ferror(f); fclose(f); require(ok,"cannot hash "+path);
    return hash.finish();
}
class File {
    FILE* file_;
    bool writing_;
    SHA256 hash_;
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
        hash_.update(data,size);
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
    Digest finish() {
        Digest digest=hash_.finish(), saved=digest;
        size_t n=writing_ ? fwrite(saved.data(),1,saved.size(),file_) : fread(saved.data(),1,saved.size(),file_);
        require(n==saved.size() && saved==digest,"SHA-256 mismatch in "+path_);
        if (writing_) require(fflush(file_)==0 && fsync(fileno(file_))==0,"cannot sync "+path_);
        else require(fgetc(file_)==EOF && !ferror(file_),"trailing data in "+path_);
        int status=fclose(file_); file_=nullptr;
        require(status==0,"cannot close "+path_); return digest;
    }
};
} // namespace puf_checkpoint
#endif
