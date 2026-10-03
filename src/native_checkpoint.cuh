#ifndef PUFFER_NATIVE_CHECKPOINT_CUH
#define PUFFER_NATIVE_CHECKPOINT_CUH

// Include after PuffeRL and the CPU worker implementation. Version 1 supports
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
inline void supported(PuffeRL* p) {
#ifndef PUF_ENV_STATE
    require(false,"this environment does not implement state save/validate/load");
#endif
    const uint32_t endian=1;
    require(*reinterpret_cast<const unsigned char*>(&endian)==1 && sizeof(long)==8 &&
        sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
        "v1 requires little-endian LP64 with IEEE float32");
    require(!USE_BF16 && PUF_BACKEND==PUF_CPU && !p->hypers.async &&
        !p->hypers.cudagraphs && p->num_policies==1,
        "v1 requires FP32, CPU env, async=0, cudagraphs=-1, and one policy");
    for (int i=0;i<p->vec->buffers;++i)
        require(__atomic_load_n(&p->vec->worker_state[i],__ATOMIC_SEQ_CST)==BUF_WAITING,
            "environment workers are not at a rollout boundary");
    cuda_ok(cudaDeviceSynchronize());
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
inline std::string identity(PuffeRL* p, Ini* ini) {
    std::vector<std::string> fields;
    for(int s=0;s<ini->num_sections;++s) {
        Dict* section=&ini->sections[s]; std::string name=section->name;
        if(name!="env" && name!="policy" && name!="train" && name!="vec" && name!="base") continue;
        for(int i=0;i<section->size;++i) {
            const DictItem& item=section->items[i]; std::string key=item.key;
            if(name=="base" && key!="seed" && key!="async" && key!="cudagraphs" &&
                key!="reset_every_horizon") continue;
#ifdef PUFFER_DECISION_POLICY
            if(name=="policy" && key=="bundle") continue; // Content identity below permits relocation.
#endif
            // Length-prefix raw values to avoid ambiguous embedded delimiters.
            std::string value=item.str ? item.str : std::to_string(item.value);
            fields.push_back(name+"."+key+"="+std::to_string(value.size())+":"+value);
        }
    }
    std::sort(fields.begin(),fields.end());
    std::ostringstream out;
    out<<"puffer-training-v1\n"<<PUFFER_ENV_NAME<<"\n";
    for(const auto& field:fields) out<<field<<"\n";
    // Raw CUDA RNG records are deliberately tied to this executable/backend.
    static const std::string executable=hex(hash_file("/proc/self/exe"));
    int runtime=0,driver=0,curand=0,nccl=0;
    cuda_ok(cudaRuntimeGetVersion(&runtime)); cuda_ok(cudaDriverGetVersion(&driver));
    require(curandGetVersion(&curand)==CURAND_STATUS_SUCCESS,"cannot read cuRAND version");
    require(ncclGetVersion(&nccl)==ncclSuccess,"cannot read NCCL version");
    cudaDeviceProp device{}; cuda_ok(cudaGetDeviceProperties(&device,p->hypers.gpu_id));
    out<<executable<<"\n"<<runtime<<":"<<driver<<":"<<curand<<":"<<nccl<<":"<<CUBLAS_VERSION
       <<":"<<sizeof(curandStatePhilox4_32_10_t)<<"\n"<<device.name<<":"<<device.major<<":"<<device.minor<<"\n";
    out<<p->hypers.world_size<<":"<<numel(p->policies[0].master_weights.shape)<<":"
       <<p->vec->size<<":"<<sizeof(obs_t)<<":"<<OBS_SIZE<<"\n";
#ifdef PUFFER_DECISION_POLICY
    const auto& root=decision_policy_context->path;
    auto manifest=pretrained::json::read(root+"/manifest.json");
    out<<"manifest:"<<hex(hash_file(root+"/manifest.json"))<<"\n";
    for(const char* key:{"encoder_config","decision_config","tokenizer","tokenizer_config"}) {
        const auto* entry=pretrained::json::optional(manifest.get(),key);
        if(entry) out<<key<<":"<<hex(hash_file(pretrained::bundle_detail::sibling(root,
            pretrained::json::as_string(entry))))<<"\n";
    }
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
    tensor("muon.momentum",p->muon.mb,true);
    add("optimizer.lr",p->muon.lr,sizeof(float),true,true);
    add("ppo.ent_coef",p->ppo_bufs.ent_coef,sizeof(float),true,true);
    add("logging.losses",p->losses,NUM_LOSSES*sizeof(float),true,true);
    add("sampler.seed",&p->seed,sizeof(p->seed),false,false);
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
    std::string id,identity;
    uint64_t epoch=0,step=0;
    std::vector<Digest> ranks;
};
inline std::string rank_path(const std::string& dir,int rank) {
    return dir+"/rank-"+std::to_string(rank)+".state";
}
inline void manifest_io(const std::string& dir,Manifest& m,bool write,int world) {
    File file(dir+"/COMMITTED",write);
    file.expect(std::string("PUFFER-TRAINING-COMMIT-1"));
    file.expect(uint64_t(world));
    m.id=file.string(m.id,64);
    m.identity=file.string(m.identity);
    m.epoch=file.number(m.epoch); m.step=file.number(m.step);
    m.ranks.resize(world);
    for(auto& digest:m.ranks) file.bytes(digest.data(),digest.size());
    file.finish();
}
inline Digest rank_io(const std::string& dir,int rank,const Manifest& m,
        std::vector<Buffer>& state,bool write) {
    File file(rank_path(dir,rank),write);
    file.expect(std::string("PUFFER-TRAINING-RANK-1"));
    file.expect(m.id); file.expect(m.identity); file.expect(uint64_t(rank));
    file.expect(m.epoch); file.expect(m.step); file.expect(uint64_t(state.size()));
    for(auto& buffer:state) {
        file.expect(buffer.name); file.expect(uint64_t(buffer.bytes));
        if(!write) buffer.saved.resize(buffer.bytes);
        file.bytes(buffer.saved.data(),buffer.saved.size());
        finite(buffer);
    }
    return file.finish();
}
inline void save(PuffeRL* p,Ini* ini,const std::string& destination) {
    supported(p);
    Manifest m;
    m.identity=identity(p,ini); m.epoch=p->epoch; m.step=p->global_step;
    m.ranks.resize(p->hypers.world_size);
    std::string staging=destination+".partial";
    if(p->hypers.rank==0) {
        struct stat info{};
        require(lstat(destination.c_str(),&info)!=0 && errno==ENOENT,
            "destination already exists: "+destination);
        require(mkdir(staging.c_str(),0700)==0,"cannot create exclusive staging directory "+staging);
        unsigned char nonce[32];
        require(getrandom(nonce,sizeof(nonce),0)==sizeof(nonce),"cannot create checkpoint ID");
        FILE* f=fopen((staging+"/ID").c_str(),"wbx");
        require(f!=nullptr,"cannot create ID");
        bool ok=fwrite(nonce,1,sizeof(nonce),f)==sizeof(nonce) && fflush(f)==0 && fsync(fileno(f))==0;
        int status=fclose(f); require(ok && status==0,"cannot write ID");
    }
    barrier(p);
    m.id=hex(hash_file(staging+"/ID"));
    auto state=buffers(p);
    int env_index=0;
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
    m.ranks[rank]=rank_io(staging,rank,m,state,true);
    state.clear();
    barrier(p);
    if(rank==0) {
        for(int r=0;r<p->hypers.world_size;++r) {
            {
                File header(rank_path(staging,r),false);
                header.expect(std::string("PUFFER-TRAINING-RANK-1"));
                header.expect(m.id); header.expect(m.identity); header.expect(uint64_t(r));
                header.expect(m.epoch); header.expect(m.step);
            }
            // A complete rank ends in its payload digest. Every rank has
            // finished/fsynced its file before this commit is published.
            FILE* f=fopen(rank_path(staging,r).c_str(),"rb");
            require(f!=nullptr,"missing rank at commit");
            bool ok=fseek(f,-32,SEEK_END)==0 && fread(m.ranks[r].data(),1,32,f)==32;
            fclose(f); require(ok,"missing rank checksum at commit");
        }
        manifest_io(staging,m,true,p->hypers.world_size);
        sync_dir(staging);
        // NFS need not implement renameat2(RENAME_NOREPLACE). Reserve the final
        // directory exclusively, then publish the already-fsynced manifest by
        // an atomic, non-replacing hard link only after every rank is in place.
        // Readers admit COMMITTED, not mere directory existence.
        require(mkdir(destination.c_str(),0700)==0,"cannot reserve destination "+destination);
        for(int r=0;r<p->hypers.world_size;++r)
            require(rename(rank_path(staging,r).c_str(),rank_path(destination,r).c_str())==0,
                "cannot move completed rank into destination");
        require(rename((staging+"/ID").c_str(),(destination+"/ID").c_str())==0,"cannot move checkpoint ID");
        sync_dir(destination);
        auto slash=destination.find_last_of('/');
        std::string parent=slash==std::string::npos ? "." : slash==0 ? "/" : destination.substr(0,slash);
        sync_dir(parent);
        require(link((staging+"/COMMITTED").c_str(),(destination+"/COMMITTED").c_str())==0,
            "cannot atomically publish commit manifest");
        sync_dir(destination);
        require(unlink((staging+"/COMMITTED").c_str())==0 && rmdir(staging.c_str())==0,
            "committed successfully but cannot remove owned staging directory");
        sync_dir(parent);
    }
    barrier(p);
}
inline void load(PuffeRL* p,Ini* ini,const std::string& source) {
    supported(p);
    char* resolved=realpath(source.c_str(),nullptr);
    require(resolved!=nullptr,"cannot resolve checkpoint directory");
    std::string canonical(resolved); free(resolved);
    require(canonical.size()<8 || canonical.substr(canonical.size()-8)!=".partial",
        "cannot resume an unpublished directory");
    Manifest m; manifest_io(source,m,false,p->hypers.world_size);
    require(m.identity==identity(p,ini),"model, config, world size, executable, or CUDA backend differs");
    const uint64_t batch=uint64_t(p->hypers.total_agents)*p->hypers.horizon;
    require(m.epoch>0 && m.epoch<=uint64_t(p->hypers.total_timesteps)/p->hypers.world_size/batch &&
        m.step==m.epoch*batch && m.step<=uint64_t(LONG_MAX),"invalid training counters");
    auto state=buffers(p);
    require(rank_io(source,p->hypers.rank,m,state,false)==m.ranks[p->hypers.rank],
        "rank does not belong to this committed checkpoint");
    int env_index=0;
#ifdef PUF_ENV_STATE
    for(const auto& buffer:state) if(!buffer.pointer)
        require(puf_state_validate(&p->vec->envs[env_index++],buffer.saved.data(),buffer.bytes)==0,
            "invalid "+buffer.name);
#endif
    // Every rank validates everything before any rank mutates live state.
    barrier(p);
    env_index=0;
    for(const auto& buffer:state) {
        if(buffer.device) cuda_ok(cudaMemcpy(buffer.pointer,buffer.saved.data(),buffer.bytes,cudaMemcpyHostToDevice));
        else if(buffer.pointer) memcpy(buffer.pointer,buffer.saved.data(),buffer.bytes);
#ifdef PUF_ENV_STATE
        else require(puf_state_load(&p->vec->envs[env_index++],buffer.saved.data(),buffer.bytes)==0,
            "cannot restore validated "+buffer.name);
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
