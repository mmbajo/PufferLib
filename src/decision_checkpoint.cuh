#pragma once

#include "decision_transformer.cuh"
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace decision {
// Little-endian v1: magic[8], width/layers/head/relative/nparams (u32), then
// named tensors: name_bytes/u32 + name, ndim/u32 + dims/u32[], count/u64,
// contiguous row-major float32 values. Names and shapes must match exactly.
namespace checkpoint {
inline void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> void write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    check((bool)out, "checkpoint write failed");
}
template<class T> T read(std::istream& in) {
    T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(value));
    check((bool)in, "truncated checkpoint"); return value;
}
inline void little_endian() {
    uint32_t one=1;
    check(*reinterpret_cast<unsigned char*>(&one)==1,"checkpoint requires little-endian host");
}
inline Config header(std::istream& in, uint32_t* count=nullptr) {
    little_endian();
    char magic[8]; in.read(magic,8);
    check((bool)in && std::string(magic,8)==std::string("PUFDT01\0",8),"invalid model checkpoint magic");
    Config c;
    uint32_t width=read<uint32_t>(in), layers=read<uint32_t>(in);
    uint32_t head=read<uint32_t>(in), relative=read<uint32_t>(in);
    uint32_t n=read<uint32_t>(in);
    check(width>=4 && width<=4096 && width%4==0 && layers>=1 && layers<=128
        && head<=1 && relative<=1 && n<=4096,"invalid model checkpoint configuration");
    c.width=(int)width; c.layers=(int)layers;
    c.head_pooling=head; c.relative_coordinates=relative;
    if (count) *count=n;
    return c;
}
inline Config config(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    check((bool)in,"cannot open model checkpoint"); return header(in);
}
inline void save(Model& model, const std::string& path) {
    little_endian();
    const auto& c=model.config();
    check(c.width>=4 && c.width<=4096 && c.width%4==0 && c.layers>=1 && c.layers<=128
        && model.parameters().size()<=4096,"model configuration exceeds checkpoint format bounds");
    // cudaMemcpy on the default stream does not order work submitted to a
    // nonblocking model stream. Save the completed parameter revision.
    check(cudaStreamSynchronize(model.stream())==cudaSuccess,"checkpoint stream synchronization failed");
    std::string temporary=path+".tmp."+std::to_string(getpid());
    try {
        std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
        check((bool)out,"cannot create model checkpoint"); out.write("PUFDT01\0",8);
        write<uint32_t>(out,c.width); write<uint32_t>(out,c.layers);
        write<uint32_t>(out,c.head_pooling); write<uint32_t>(out,c.relative_coordinates);
        write<uint32_t>(out,(uint32_t)model.parameters().size());
        for (const auto& p:model.parameters()) {
            write<uint32_t>(out,(uint32_t)p.name.size()); out.write(p.name.data(),p.name.size());
            write<uint32_t>(out,(uint32_t)p.shape.size());
            for (int dim:p.shape) write<uint32_t>(out,dim);
            write<uint64_t>(out,p.count);
            std::vector<float> values(p.count);
            auto status=cudaMemcpy(values.data(),p.data,p.count*sizeof(float),cudaMemcpyDeviceToHost);
            check(status==cudaSuccess,"checkpoint device copy failed");
            out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
        }
        out.flush(); check((bool)out,"checkpoint flush failed"); out.close();
        check(!out.fail(),"checkpoint close failed");
        check(std::rename(temporary.c_str(),path.c_str())==0,"checkpoint rename failed");
    } catch (...) { std::remove(temporary.c_str()); throw; }
}
inline void load(Model& model, const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    check((bool)in,"cannot open model checkpoint");
    uint32_t n; auto c=header(in,&n), expected=model.config();
    check(c.width==expected.width && c.layers==expected.layers
        && c.head_pooling==expected.head_pooling
        && c.relative_coordinates==expected.relative_coordinates
        && n==model.parameters().size(),"model checkpoint architecture mismatch");
    // Validate every tensor before changing any device weights.
    std::vector<std::vector<float>> tensors;
    for (const auto& p:model.parameters()) {
        uint32_t len=read<uint32_t>(in);
        check(len==p.name.size(),"model checkpoint parameter name length mismatch");
        std::string name(len,'\0'); in.read(&name[0],len);
        check(name==p.name,"model checkpoint parameter name mismatch");
        check(read<uint32_t>(in)==p.shape.size(),"model checkpoint tensor rank mismatch");
        for (int dim:p.shape) check(read<uint32_t>(in)==(uint32_t)dim,"model checkpoint shape mismatch");
        check(read<uint64_t>(in)==p.count,"model checkpoint count mismatch");
        tensors.emplace_back(p.count);
        in.read(reinterpret_cast<char*>(tensors.back().data()),p.count*sizeof(float));
        check((bool)in,"truncated model checkpoint tensor");
        for (float value:tensors.back()) check(std::isfinite(value),"nonfinite checkpoint weight");
    }
    check(in.peek()==std::char_traits<char>::eof(),"trailing model checkpoint data");
    // Finish pending uses or updates before replacing the parameter storage.
    // Upload on the model stream and wait before host staging tensors disappear.
    check(cudaStreamSynchronize(model.stream())==cudaSuccess,"checkpoint stream synchronization failed");
    for (size_t i=0;i<tensors.size();++i)
        check(cudaMemcpyAsync(model.parameters()[i].data,tensors[i].data(),tensors[i].size()*sizeof(float),
            cudaMemcpyHostToDevice,model.stream())==cudaSuccess,"checkpoint upload failed");
    check(cudaStreamSynchronize(model.stream())==cudaSuccess,"checkpoint upload synchronization failed");
}
}
}
