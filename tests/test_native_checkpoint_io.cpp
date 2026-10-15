// Standalone CPU record tests; see tools/native_training_checkpoints.md.
#include "../src/native_checkpoint_io.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <sys/stat.h>

using puf_checkpoint::File;
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

static const std::string marker = "PUFFER-TRAINING-RANK";
static const std::string embedded("alpha\0beta\n=:",13);
static Bytes payload() {
    Bytes data(65536+37);
    for(size_t i=0;i<data.size();++i) data[i]=(i*29+(i>>8)*17+53)&255;
    return data;
}
static void save(const std::string& path, uint64_t version=2,
        const std::string& name="weights", int length_delta=0) {
    File file(path,true);
    file.expect(marker+"-"+std::to_string(version));
    file.number(0); file.number(UINT64_C(0x0102030405060708)); file.number(UINT64_MAX);
    file.string(""); file.string(embedded,embedded.size());
    file.expect(name);
    auto data=payload(); file.number(data.size()+length_delta);
    file.bytes(data.data(),19); file.bytes(data.data()+19,data.size()-19);
    file.finish();
}
static Bytes load(const std::string& path) {
    File file(path,false);
    file.record(marker);
    assert(file.number()==0);
    assert(file.number()==UINT64_C(0x0102030405060708));
    assert(file.number()==UINT64_MAX);
    assert(file.string("",0).empty());
    assert(file.string("",embedded.size())==embedded);
    file.expect(std::string("weights"));
    auto expected=payload(); file.expect(uint64_t(expected.size()));
    Bytes actual(expected.size());
    file.bytes(actual.data(),actual.size());
    file.finish();
    return actual;
}

static void containers() {
    Temporary directory;
    const auto original=directory.file("original"), changed=directory.file("changed");
    save(original);
    assert(load(original)==payload());
    Bytes bytes=read(original);
    // Assert the actual wire encoding, including unaligned integer positions.
    assert(bytes[0]==marker.size()+2);
    for(size_t i=1;i<8;++i) assert(bytes[i]==0);
    size_t number=8+marker.size()+2+8;
    for(size_t i=0;i<8;++i) assert(bytes[number+i]==8-i);
    const auto before=read(original);
    rejects([&]{File duplicate(original,true);},"cannot open");
    assert(read(original)==before);
    rejects([&]{File missing(directory.path+"/missing",false);},"cannot open");

    for(size_t n=0;n<160;++n) {
        write(changed,Bytes(bytes.begin(),bytes.begin()+n));
        rejects([&]{load(changed);},"short IO");
    }
    for(size_t n : {size_t(65535),size_t(65536),bytes.size()-1}) {
        write(changed,Bytes(bytes.begin(),bytes.begin()+n));
        rejects([&]{load(changed);},"short IO");
    }
    auto altered=bytes;
    altered.back()^=1;
    write(changed,altered);
    auto expected=payload(); expected.back()^=1;
    assert(load(changed)==expected); // Raw payload bytes are carried directly.
    for(size_t count : {size_t(1),size_t(32),size_t(4096)}) {
        auto bad=bytes; bad.insert(bad.end(),count,0); write(changed,bad);
        rejects([&]{load(changed);},"trailing data");
    }
    const auto old_version=directory.file("old-version"); save(old_version,1);
    rejects([&]{load(old_version);},"feat/typed-supervision at 148add30");
    const auto wrong_version=directory.file("wrong-version"); save(wrong_version,3);
    rejects([&]{load(wrong_version);},"unsupported record version");
    const auto wrong_name=directory.file("wrong-name"); save(wrong_name,2,"actions");
    rejects([&]{load(wrong_name);},"incompatible metadata");
    const auto wrong_length=directory.file("wrong-length"); save(wrong_length,2,"weights",1);
    rejects([&]{load(wrong_length);},"incompatible integer");
    const auto oversized=directory.file("oversized");
    {File file(oversized,true);file.number(UINT64_MAX);file.finish();}
    rejects([&]{File file(oversized,false);file.string("",1024);},"oversized string");
    const auto over_write=directory.file("oversized-write");
    rejects([&]{File file(over_write,true);file.string("abc",2);},"oversized string");
    const auto abandoned=directory.file("abandoned");
    {File file(abandoned,true);file.record(marker);}
    rejects([&]{load(abandoned);},"short IO");
}

int main() {
    try {
        containers();
        puts("Native checkpoint IO tests passed: v2 wire encoding, old-version refusal, "
            "bounds, direct payload bytes, truncation, trailing data, exclusive creation");
    } catch(const std::exception& error) {
        fprintf(stderr,"%s\n",error.what());
        return 1;
    }
    return 0;
}
