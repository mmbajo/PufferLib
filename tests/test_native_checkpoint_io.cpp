// Standalone CPU container/SHA checks:
// c++ -std=c++17 -O2 -Wall -Wextra -Werror tests/test_native_checkpoint_io.cpp -o build/test_native_checkpoint_io
// build/test_native_checkpoint_io
// build/test_native_checkpoint_io --hashlib  # Optional independent Python oracle.
// --hash FILE and --stream FILE CHUNK support independent hashlib crosschecks.
#include "../src/native_checkpoint_io.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <sys/stat.h>

using puf_checkpoint::Digest;
using puf_checkpoint::File;
using puf_checkpoint::SHA256;
using Bytes = std::vector<unsigned char>;

struct Temporary {
    std::string path;
    std::vector<std::string> files;
    Temporary() {
        char pattern[] = "/tmp/puffer-checkpoint-io-XXXXXX";
        char* result = mkdtemp(pattern);
        assert(result); path = result;
    }
    std::string file(const std::string& name) {
        std::string result = path + "/" + name;
        files.push_back(result); return result;
    }
    ~Temporary() {
        for (const auto& file : files) unlink(file.c_str());
        rmdir(path.c_str());
    }
};

template<class F> static void rejects(F operation, const char* message) {
    bool rejected = false;
    try { operation(); }
    catch (const std::runtime_error& error) {
        rejected = true;
        assert(std::string(error.what()).find(message) != std::string::npos);
    }
    assert(rejected);
}

static Bytes read(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return Bytes(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
static void write(const std::string& path, const Bytes& data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    assert(file);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close(); assert(file);
}

static Digest stream_hash(const std::string& path, size_t chunk) {
    assert(chunk > 0 && chunk <= 16*1024*1024);
    std::ifstream file(path, std::ios::binary);
    assert(file); Bytes buffer(chunk); SHA256 hash;
    do {
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        hash.update(buffer.data(), size_t(file.gcount()));
    } while (file);
    assert(file.eof()); return hash.finish();
}

static void check_sha(const std::string& input, const char* expected) {
    for (size_t chunk : {size_t(1),size_t(7),size_t(55),size_t(56),size_t(63),
            size_t(64),size_t(65),size_t(4096),size_t(65536)}) {
        SHA256 hash;
        hash.update(nullptr, 0);
        for (size_t at = 0; at < input.size(); at += chunk)
            hash.update(input.data()+at, std::min(chunk,input.size()-at));
        assert(puf_checkpoint::hex(hash.finish()) == expected);
        assert(puf_checkpoint::hex(hash.finish()) == expected);
    }
}

static void sha_vectors() {
    check_sha("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check_sha("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    check_sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    check_sha("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
        "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    check_sha(std::string(1000000,'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    SHA256 incremental;
    incremental.update("a",1);
    assert(puf_checkpoint::hex(incremental.finish()) ==
        "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb");
    incremental.update("bc",2); // finish() must not consume mutable state.
    assert(puf_checkpoint::hex(incremental.finish()) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

static const std::string marker = "PUFFER-HOST-CONTAINER-TEST-1";
static const std::string embedded("alpha\0beta\n=:",13);
static Bytes payload() {
    Bytes data(65536+37);
    for(size_t i=0;i<data.size();++i) data[i]=(i*29+(i>>8)*17+53)&255;
    return data;
}
static Digest save(const std::string& path, uint64_t version=1,
        const std::string& name="weights", int length_delta=0) {
    File file(path,true);
    file.expect(marker); file.expect(version);
    file.number(0); file.number(UINT64_C(0x0102030405060708)); file.number(UINT64_MAX);
    file.string(""); file.string(embedded,embedded.size());
    file.expect(name);
    auto data=payload(); file.number(data.size()+length_delta);
    file.bytes(data.data(),19); file.bytes(data.data()+19,data.size()-19);
    return file.finish();
}
static Digest load(const std::string& path) {
    File file(path,false);
    file.expect(marker); file.expect(uint64_t(1));
    assert(file.number()==0);
    assert(file.number()==UINT64_C(0x0102030405060708));
    assert(file.number()==UINT64_MAX);
    assert(file.string("",0).empty());
    assert(file.string("",embedded.size())==embedded);
    file.expect(std::string("weights"));
    auto expected=payload(); file.expect(uint64_t(expected.size()));
    Bytes actual(expected.size());
    file.bytes(actual.data(),actual.size());
    auto digest=file.finish();
    assert(actual==expected); return digest;
}

static void containers() {
    Temporary directory;
    const auto original=directory.file("original"), changed=directory.file("changed");
    const Digest saved=save(original);
    assert(load(original)==saved);
    Bytes bytes=read(original);
    assert(bytes.size()>65536);
    // The trailer hashes every payload byte, but does not hash itself.
    SHA256 independent; independent.update(bytes.data(),bytes.size()-32);
    assert(independent.finish()==saved);
    assert(std::equal(saved.begin(),saved.end(),bytes.end()-32));
    assert(stream_hash(original,1)==puf_checkpoint::hash_file(original));
    assert(stream_hash(original,65537)==puf_checkpoint::hash_file(original));
    // Assert the actual wire encoding, including unaligned integer positions.
    assert(bytes[0]==marker.size());
    for(size_t i=1;i<8;++i) assert(bytes[i]==0);
    size_t number=8+marker.size()+8+8;
    for(size_t i=0;i<8;++i) assert(bytes[number+i]==8-i);
    const auto before=read(original);
    rejects([&]{File duplicate(original,true);},"cannot open");
    assert(read(original)==before);
    rejects([&]{File missing(directory.path+"/missing",false);},"cannot open");
    rejects([&]{puf_checkpoint::hash_file(directory.path+"/missing");},"cannot open");
    rejects([&]{puf_checkpoint::hash_file(directory.path);},"cannot hash");

    for(size_t n=0;n<160;++n) {
        write(changed,Bytes(bytes.begin(),bytes.begin()+n));
        rejects([&]{load(changed);},"short IO");
    }
    for(size_t n : {size_t(65535),size_t(65536),bytes.size()-33,bytes.size()-32,
            bytes.size()-31,bytes.size()-1}) {
        write(changed,Bytes(bytes.begin(),bytes.begin()+n));
        rejects([&]{load(changed);}, n>=bytes.size()-32 ? "SHA-256 mismatch" : "short IO");
    }
    for(size_t offset : {size_t(180),size_t(255),size_t(65535),size_t(65536),bytes.size()-33}) {
        auto bad=bytes; bad[offset]^=0x80; write(changed,bad);
        rejects([&]{load(changed);},"SHA-256 mismatch");
    }
    for(size_t offset=bytes.size()-32;offset<bytes.size();++offset) {
        auto bad=bytes; bad[offset]^=1; write(changed,bad);
        rejects([&]{load(changed);},"SHA-256 mismatch");
    }
    for(size_t count : {size_t(1),size_t(32),size_t(4096)}) {
        auto bad=bytes; bad.insert(bad.end(),count,0); write(changed,bad);
        rejects([&]{load(changed);},"trailing data");
    }
    const auto wrong_version=directory.file("wrong-version"); save(wrong_version,2);
    rejects([&]{load(wrong_version);},"incompatible integer");
    const auto wrong_name=directory.file("wrong-name"); save(wrong_name,1,"actions");
    rejects([&]{load(wrong_name);},"incompatible metadata");
    const auto wrong_length=directory.file("wrong-length"); save(wrong_length,1,"weights",1);
    rejects([&]{load(wrong_length);},"incompatible integer");
    const auto oversized=directory.file("oversized");
    {File file(oversized,true);file.number(UINT64_MAX);file.finish();}
    rejects([&]{File file(oversized,false);file.string("",1024);},"oversized string");
    const auto over_write=directory.file("oversized-write");
    rejects([&]{File file(over_write,true);file.string("abc",2);},"oversized string");
    const auto abandoned=directory.file("abandoned");
    {File file(abandoned,true);file.expect(marker);file.expect(uint64_t(1));}
    rejects([&]{load(abandoned);},"short IO");
}

int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--hashlib") {
            // Separate implementation, including the 2^32-bit message-length
            // boundary. Arguments go through exec directly, never a shell.
            const char* script=R"PY(
import hashlib, subprocess, sys, tempfile
from pathlib import Path
binary=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='checkpoint-sha-') as directory:
    path=Path(directory)/'payload'
    block=bytes((i*29+(i>>8)*17+53)&255 for i in range(1<<20))
    oracle=hashlib.sha256()
    with path.open('wb') as stream:
        for _ in range(513):
            stream.write(block); oracle.update(block)
        stream.write(block[:19]); oracle.update(block[:19])
    expected=oracle.hexdigest()
    for arguments in (['--hash',str(path)],['--stream',str(path),'65537']):
        actual=subprocess.check_output([str(binary),*arguments],text=True).strip()
        assert actual==expected,(arguments,actual,expected)
    print(f'hashlib PASS: {path.stat().st_size} bytes (>2^32 bits), SHA256 {expected}')
    for size in (0,1,54,55,56,57,63,64,65,119,120,127,128,129,65535,65536,65537):
        path.write_bytes(block[:size])
        expected=hashlib.sha256(block[:size]).hexdigest()
        actual=subprocess.check_output([str(binary),'--hash',str(path)],text=True).strip()
        assert actual==expected,(size,actual,expected)
    print('hashlib padding/block-boundary checks PASS: 17 lengths')
)PY";
            execlp("python3","python3","-c",script,argv[0],static_cast<char*>(nullptr));
            perror("python3"); return 1;
        }
        if(argc==3 && std::string(argv[1])=="--hash") {
            puts(puf_checkpoint::hex(puf_checkpoint::hash_file(argv[2])).c_str());return 0;
        }
        if(argc==4 && std::string(argv[1])=="--stream") {
            char* end=nullptr; errno=0;
            unsigned long long chunk=strtoull(argv[3],&end,10);
            if(errno || !end || *end || !chunk || chunk>16*1024*1024) return 2;
            puts(puf_checkpoint::hex(stream_hash(argv[2],size_t(chunk))).c_str());return 0;
        }
        if(argc!=1) {fprintf(stderr,"usage: %s [--hashlib | --hash FILE | --stream FILE CHUNK]\n",argv[0]);return 2;}
        sha_vectors(); containers();
        puts("Native checkpoint IO tests passed: SHA vectors, wire encoding, bounds, corruption, truncation, exclusive creation");
    } catch(const std::exception& error) {fprintf(stderr,"%s\n",error.what());return 1;}
    return 0;
}
