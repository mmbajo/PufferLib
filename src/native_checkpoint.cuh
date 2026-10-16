#ifndef PUFFER_NATIVE_CHECKPOINT_CUH
#define PUFFER_NATIVE_CHECKPOINT_CUH

// Include after PuffeRL and the CPU worker implementation. Version 2 supports
// synchronous FP32 training with a single policy and an explicit env codec.
#include "native_checkpoint_io.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/random.h>

namespace puf_checkpoint {
inline void cuda_ok(cudaError_t status) { require(status==cudaSuccess,cudaGetErrorString(status)); }
inline long integer_option(Ini* ini,const char* key,long maximum) {
    auto* item=dict_find(puf_ini_section(ini,"base",0),key);
    double value=0;
    require(item && item->str && item->len==0 && puf_ini_parse_val(item->str,&value) &&
        std::isfinite(value) && value>=0 && value<=double(maximum) && std::floor(value)==value,
        std::string("base.")+key+" must be a bounded nonnegative integer");
    return long(value);
}
inline void supported() {
#ifndef PUF_ENV_STATE
    require(false,"this environment does not implement state save/validate/load");
#endif
    require(!USE_BF16,"full state requires FP32");
}
inline void barrier(PuffeRL* p) {
    if (p->hypers.world_size==1) return;
    cuda_ok(cudaMemset(p->dp_reduce_scratch,0,sizeof(double)));
    require(ncclAllReduce(p->dp_reduce_scratch,p->dp_reduce_scratch,1,ncclDouble,
        ncclSum,p->nccl_comm,p->default_stream)==ncclSuccess,"checkpoint barrier failed");
    cuda_ok(cudaStreamSynchronize(p->default_stream));
}
inline void sync_dir(const std::string& path) {
    int fd=open(path.c_str(),O_RDONLY|O_DIRECTORY);
    require(fd>=0,"cannot open directory "+path);
    int status=fsync(fd); close(fd); require(status==0,"cannot sync directory "+path);
}
inline std::string metadata(PuffeRL* p, Ini* ini) {
    std::vector<std::string> fields;
    for(int s=0;s<ini->num_sections;++s) {
        Dict* section=&ini->sections[s]; std::string name=section->name;
        if(name!="env" && name!="policy" && name!="train" && name!="vec" && name!="base") continue;
        for(int i=0;i<section->size;++i) {
            const DictItem& item=section->items[i]; std::string key=item.key;
            if(name=="base" && key!="seed" && key!="async" && key!="cudagraphs" &&
                key!="reset_every_horizon") continue;
            // Length-prefix raw values to avoid ambiguous embedded delimiters.
            std::string value=item.str ? item.str : std::to_string(item.value);
#ifdef PUFFER_DECISION_POLICY
            if(name=="policy" && key=="bundle") {
                char* path=realpath(value.c_str(),nullptr);
                require(path!=nullptr,"cannot resolve bundle path");
                value=path; free(path);
            }
#endif
            fields.push_back(name+"."+key+"="+std::to_string(value.size())+":"+value);
        }
    }
    std::sort(fields.begin(),fields.end());
    std::ostringstream out;
    out<<std::setprecision(std::numeric_limits<float>::max_digits10);
    out<<"puffer-training-v2\n"<<PUFFER_ENV_NAME<<"\n";
    for(const auto& field:fields) out<<field<<"\n";
    // Keep the producing binary at this path: CUDA RNG records use its ABI.
    char* executable=realpath("/proc/self/exe",nullptr);
    require(executable!=nullptr,"cannot resolve executable path");
    out<<executable<<"\n"<<sizeof(curandStatePhilox4_32_10_t)<<"\n";
    out<<p->hypers.world_size<<":"<<numel(p->policies[0].master_weights.shape)<<":"
       <<p->vec->size<<":"<<sizeof(obs_t)<<":"<<OBS_SIZE<<"\n";
    free(executable);
#ifdef PUFFER_DECISION_POLICY
    const auto& context=*decision_policy_context;
    const auto& config=context.config;
    const auto& encoder=config.encoder;
    const auto& bundle=context.bundle;
    out<<"encoder:"<<encoder.family<<":"<<encoder.width<<":"<<encoder.layers<<":"
        <<encoder.heads<<":"<<encoder.intermediate<<":"<<encoder.vocab<<":"
        <<encoder.max_positions<<":"<<encoder.type_vocab<<":"<<encoder.pad_token_id<<":"
        <<encoder.epsilon<<":"<<encoder.attention_bias<<":"<<encoder.mlp_bias<<":"
        <<encoder.norm_bias<<":"<<encoder.local_window<<":"<<encoder.global_every<<":"
        <<encoder.global_rope_theta<<":"<<encoder.local_rope_theta<<":"
        <<pretrained::gemm_precision_name(encoder.gemm_precision)<<"\n";
    out<<"layers:";
    for(int type:encoder.layer_types) out<<type<<",";
    out<<"\ndecision:"<<config.head_layers<<":"<<config.n_act<<":"<<config.value_head<<":"
        <<context.execution_tokens<<":"<<context.temperature<<"\n";
    out<<"packing:"<<bundle.kind<<":"<<bundle.max_len<<":"<<bundle.head_max_len<<":"
        <<bundle.cls_id<<":"<<bundle.sep_id<<":"<<bundle.mask_id<<":"<<bundle.pad_id<<"\n";
    for(const auto& token:{bundle.cls_token,bundle.sep_token,bundle.mask_token,bundle.pad_token})
        out<<token.size()<<":"<<token<<"\n";
#endif
    return out.str();
}
struct Buffer {
    std::string name;
    void* pointer;
    size_t bytes;
    bool device, floats;
    std::vector<unsigned char> saved;
};
inline std::vector<Buffer> buffers(PuffeRL* p) {
    std::vector<Buffer> result;
    auto add=[&](const std::string& name,void* ptr,size_t size,bool device,bool floats) {
        require(ptr!=nullptr || size==0,"missing buffer "+name);
        result.push_back({name,ptr,size,device,floats,{}});
    };
    auto tensor=[&](const std::string& name,auto t,bool floats) {
        add(name,t.data,numel(t.shape)*sizeof(*t.data),true,floats);
    };
    tensor("weights",p->policies[0].master_weights,true);
    if(p->hypers.optimizer==PUF_OPTIMIZER_ADAM) {
        tensor("adam.first_moment",p->adam.first_moment,true);
        tensor("adam.second_moment",p->adam.second_moment,true);
        add("adam.step",p->adam.step,sizeof(uint64_t),true,false);
    } else tensor("muon.momentum",p->muon.mb,true);
    add("logging.losses",p->losses,NUM_LOSSES*sizeof(float),true,true);
    add("rng.offset",p->rng_offset,(p->vec->buffers+1)*sizeof(long),true,false);
    for(int i=0;i<p->vec->buffers;++i) {
        add("rng.philox."+std::to_string(i),p->rng_states[i],
            p->vec->agents_per_buf*sizeof(curandStatePhilox4_32_10_t),true,false);
        tensor("policy.carry."+std::to_string(i),p->policies[0].buffer_states[i],true);
    }
    tensor("env.observations",p->env.obs,sizeof(obs_t)==sizeof(float));
    tensor("env.actions",p->env.actions,true);
    tensor("env.rewards",p->env.rewards,true);
    tensor("env.terminals",p->env.terminals,true);
    tensor("env.mask",p->env.action_mask,false);
#ifdef PUF_HAS_TRUNCATION
    tensor("env.final_observations",p->env.final_observations,true);
    tensor("env.truncations",p->env.truncations,false);
    add("host.final_observations",p->vec->final_observations,
        p->vec->total_agents*OBS_SIZE*sizeof(precision_t),false,true);
    add("host.truncations",p->vec->truncations,p->vec->total_agents,false,false);
#endif
#ifdef PUF_ENV_STATE
    for(int i=0;i<p->vec->size;++i)
        result.push_back({"environment."+std::to_string(i),nullptr,
            puf_state_size(&p->vec->envs[i]),false,false,{}});
#endif
    return result;
}
inline void finite(const Buffer& buffer) {
    if(!buffer.floats) return;
    require(buffer.bytes%sizeof(float)==0,"unaligned float buffer");
    for(size_t i=0;i<buffer.bytes;i+=sizeof(float)) {
        float value; memcpy(&value,buffer.saved.data()+i,sizeof(value));
        if(!std::isfinite(value)) require(false,"nonfinite state in "+buffer.name);
    }
}
struct Manifest {
    std::string id=std::string(32,'\0'), metadata;
    uint64_t epoch=0,step=0;
};
inline std::string rank_path(const std::string& dir,int rank) {
    return dir+"/rank-"+std::to_string(rank)+".state";
}
inline void manifest_io(const std::string& dir,Manifest& m,bool write,int world) {
    File file(dir+"/COMMITTED",write);
    file.record("PUFFER-TRAINING-COMMIT");
    file.expect(uint64_t(world));
    file.bytes(m.id.data(),m.id.size());
    m.metadata=file.string(m.metadata);
    m.epoch=file.number(m.epoch); m.step=file.number(m.step);
    file.finish();
}
inline void rank_io(const std::string& dir,int rank,const Manifest& m,
        std::vector<Buffer>& state,bool write) {
    File file(rank_path(dir,rank),write);
    file.record("PUFFER-TRAINING-RANK");
    file.expect(m.id); file.expect(m.metadata); file.expect(uint64_t(rank));
    file.expect(m.epoch); file.expect(m.step); file.expect(uint64_t(state.size()));
    for(auto& buffer:state) {
        file.expect(buffer.name); file.expect(uint64_t(buffer.bytes));
        if(!write) buffer.saved.resize(buffer.bytes);
        file.bytes(buffer.saved.data(),buffer.saved.size());
        if(!write) finite(buffer);
    }
    file.finish();
}
inline void save(PuffeRL* p,Ini* ini,const std::string& destination) {
    Manifest m;
    m.metadata=metadata(p,ini); m.epoch=p->epoch; m.step=p->global_step;
    std::string staging=destination+".partial";
    if(p->hypers.rank==0) {
        struct stat info{};
        require(lstat(destination.c_str(),&info)!=0 && errno==ENOENT,
            "destination already exists: "+destination);
        require(mkdir(staging.c_str(),0700)==0,"cannot create exclusive staging directory "+staging);
        require(getrandom(m.id.data(),m.id.size(),0)==ssize_t(m.id.size()),
            "cannot create checkpoint ID");
        File id(staging+"/ID",true);
        id.bytes(m.id.data(),m.id.size());
        id.finish();
    }
    barrier(p);
    File id(staging+"/ID",false);
    id.bytes(m.id.data(),m.id.size());
    id.finish();
    auto state=buffers(p);
#ifdef PUF_ENV_STATE
    int env_index=0;
#endif
    for(auto& buffer:state) {
        buffer.saved.resize(buffer.bytes);
        if(buffer.device) cuda_ok(cudaMemcpy(buffer.saved.data(),buffer.pointer,buffer.bytes,cudaMemcpyDeviceToHost));
        else if(buffer.pointer) memcpy(buffer.saved.data(),buffer.pointer,buffer.bytes);
#ifdef PUF_ENV_STATE
        else require(puf_state_save(&p->vec->envs[env_index++],buffer.saved.data(),buffer.bytes)==0,
            "cannot serialize "+buffer.name);
#endif
    }
    const int rank=p->hypers.rank;
    rank_io(staging,rank,m,state,true);
    state.clear();
    barrier(p);
    if(rank==0) {
        require(unlink((staging+"/ID").c_str())==0,"cannot remove staging checkpoint ID");
        manifest_io(staging,m,true,p->hypers.world_size);
        sync_dir(staging);
        require(rename(staging.c_str(),destination.c_str())==0,
            "cannot publish checkpoint directory");
        auto slash=destination.find_last_of('/');
        std::string parent=slash==std::string::npos ? "." : slash==0 ? "/" : destination.substr(0,slash);
        sync_dir(parent);
    }
    barrier(p);
}
inline void load(PuffeRL* p,Ini* ini,const std::string& source) {
    Manifest m; manifest_io(source,m,false,p->hypers.world_size);
    require(m.metadata==metadata(p,ini),
        "model, config, world size, or executable differs");
    const uint64_t batch=uint64_t(p->hypers.total_agents)*p->hypers.horizon;
    require(m.epoch>0 && m.epoch<=uint64_t(p->hypers.total_timesteps)/p->hypers.world_size/batch &&
        m.step==m.epoch*batch && m.step<=uint64_t(LONG_MAX),"invalid training counters");
    auto state=buffers(p);
    rank_io(source,p->hypers.rank,m,state,false);
    if(p->hypers.optimizer==PUF_OPTIMIZER_ADAM) {
        const int updates=std::max(0,int(p->hypers.replay_ratio*int(batch)/p->hypers.minibatch_size));
        require(updates==0 || m.epoch<=UINT64_MAX/uint64_t(updates),"invalid Adam update count");
        for(const auto& buffer:state) {
            if(buffer.name=="adam.step") {
                uint64_t step; memcpy(&step,buffer.saved.data(),sizeof(step));
                require(step==m.epoch*uint64_t(updates),"Adam step does not match training epoch");
            }
            if(buffer.name=="adam.second_moment")
                for(size_t i=0;i<buffer.bytes;i+=sizeof(float)) {
                    float value; memcpy(&value,buffer.saved.data()+i,sizeof(value));
                    if(value<0) require(false,"negative Adam second moment");
                }
        }
    }
#ifdef PUF_ENV_STATE
    std::vector<EnvState> decoded(p->vec->size);
    int env_index=0;
    try {
        for(const auto& buffer:state) if(!buffer.pointer) {
            require(puf_state_validate(&p->vec->envs[env_index],buffer.saved.data(),
                buffer.bytes,&decoded[env_index])==0,"invalid "+buffer.name);
            ++env_index;
        }
        // Every rank validates everything before any rank mutates live state.
        barrier(p);
    } catch(...) {
        for(auto& env:decoded) {
            puf_state_discard(&env);
        }
        throw;
    }
    env_index=0;
#else
    barrier(p);
#endif
    for(const auto& buffer:state) {
        if(buffer.device) cuda_ok(cudaMemcpy(buffer.pointer,buffer.saved.data(),buffer.bytes,cudaMemcpyHostToDevice));
        else if(buffer.pointer) memcpy(buffer.pointer,buffer.saved.data(),buffer.bytes);
#ifdef PUF_ENV_STATE
        else {
            puf_state_load(&p->vec->envs[env_index],&decoded[env_index]);
            ++env_index;
        }
#endif
    }
    p->epoch=m.epoch; p->global_step=m.step;
    p->last_log_step=p->global_step;
    p->start_time=p->last_log_time=wall_clock();
    cuda_ok(cudaDeviceSynchronize());
    barrier(p);
}
} // namespace puf_checkpoint
#endif
