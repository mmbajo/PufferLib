#include "../src/pufferl.cu"
#include <iostream>

static std::vector<puf_checkpoint::Buffer> capture(PuffeRL* p) {
    puf_checkpoint::supported(p);
    auto state=puf_checkpoint::buffers(p);
    int env=0;
    for(auto& b:state) {
        b.saved.resize(b.bytes);
        if(b.device) puf_checkpoint::cuda_ok(cudaMemcpy(b.saved.data(),b.pointer,b.bytes,cudaMemcpyDeviceToHost));
        else if(b.pointer) memcpy(b.saved.data(),b.pointer,b.bytes);
        else puf_checkpoint::require(puf_state_save(&p->vec->envs[env++],b.saved.data(),b.bytes)==0,"test capture");
    }
    return state;
}
struct CheckpointTest { Ini ini; std::string path,mode; };
static int checkpoint_worker(TrainContext* ctx,void*,void* user) {
    using namespace puf_checkpoint;
    try {
        auto& task=*static_cast<CheckpointTest*>(user);
        auto* p=create_pufferl(&task.ini,ctx);
        if(task.mode=="save") {
            for(int i=0;i<2;++i) { rollouts(p); train_impl(p,nullptr); }
            save(p,&task.ini,task.path);
        } else {
            load(p,&task.ini,task.path);
            require(p->epoch==2 && p->global_step==64,"resume counters");
        }
        Manifest m; manifest_io(task.path,m,false,ctx->world_size);
        auto expected=buffers(p);
        require(rank_io(task.path,ctx->rank,m,expected,false)==m.ranks[ctx->rank],"test input hash");
        auto live=capture(p);
        for(size_t i=0;i<live.size();++i)
            require(live[i].saved==expected[i].saved,"restore is not exact: "+live[i].name);
        // Snapshot every training state, advance through env boundaries and an
        // optimizer update, then require an exact transactional restoration.
        rollouts(p);
        SHA256 next;
        for(const auto& buffer:capture(p)) next.update(buffer.saved.data(),buffer.saved.size());
        std::cout<<"rank="<<ctx->rank<<" next_rollout_sha="<<hex(next.finish())<<"\n";
        std::vector<float> actions(numel(p->rollouts.actions.shape));
        cuda_ok(cudaMemcpy(actions.data(),p->rollouts.actions.data,actions.size()*sizeof(float),cudaMemcpyDeviceToHost));
        train_impl(p,nullptr);
        load(p,&task.ini,task.path);
        live=capture(p);
        for(size_t i=0;i<live.size();++i)
            require(live[i].saved==expected[i].saved,"mutated state was not restored: "+live[i].name);
        rollouts(p);
        std::vector<float> replay(actions.size());
        cuda_ok(cudaMemcpy(replay.data(),p->rollouts.actions.data,replay.size()*sizeof(float),cudaMemcpyDeviceToHost));
        require(actions==replay,"restored sampler/environment does not reproduce next rollout");
        train_impl(p,nullptr);
        float lr=0;
        cuda_ok(cudaMemcpy(&lr,puf_optimizer_lr(p),sizeof(lr),cudaMemcpyDeviceToHost));
        float expected_lr=cosine_annealing(p->hypers.lr,p->hypers.min_lr_ratio*p->hypers.lr,2,8);
        require(lr==expected_lr,"resume restarted learning-rate schedule");
        require(p->epoch==3 && p->global_step==96,"continued counters");
        if(p->hypers.optimizer==PUF_OPTIMIZER_ADAM) {
            uint64_t step=0; cuda_ok(cudaMemcpy(&step,p->adam.step,sizeof(step),cudaMemcpyDeviceToHost));
            require(step==6,"Adam bias-correction step did not continue");
        }
        float entropy=0;
        cuda_ok(cudaMemcpy(&entropy,p->ppo_bufs.ent_coef,sizeof(entropy),cudaMemcpyDeviceToHost));
        require(entropy==cosine_annealing(p->hypers.ent_coef,
            p->hypers.min_ent_coef_ratio*p->hypers.ent_coef,2,8),"resume restarted entropy schedule");
        if(ctx->rank==0) std::cout<<"PASS: "<<task.mode<<" exact state, next actions, optimizer, counters, schedule\n";
        env_close(p->vec);
        return 0;
    } catch(const std::exception& error) {
        fprintf(stderr,"checkpoint test rank %d: %s\n",ctx->rank,error.what()); return 1;
    }
}
int main(int argc,char** argv) {
    try {
        puf_checkpoint::require(argc>=5,"usage: test_native_checkpoint save|load PATH GPUS SHARD [BUNDLE] [--optimizer=adam|muon]");
        CheckpointTest task{}; task.mode=argv[1]; task.path=argv[2];
        puf_checkpoint::require(task.mode=="save" || task.mode=="load","invalid test mode");
        int world=atoi(argv[3]);
        puf_ini_load_env(&task.ini,PUFFER_ENV_NAME,0,nullptr);
        const char* settings[][2]={
            {"base.async","0"},{"base.cudagraphs","-1"},{"base.seed","73"},
            {"vec.total_agents","4"},{"vec.num_buffers","2"},{"vec.num_threads","1"},
            {"vec.num_policies","1"},{"vec.hist_policy_percent","0"},
            {"env.max_steps","9"},{"train.horizon","8"},{"train.minibatch_size","16"},
            {"train.learning_rate","0.001"},{"train.anneal_lr","1"},{"train.replay_ratio","1"},
            {"train.anneal_ent_coef","1"},{"base.eval_episodes","0"},
        };
        for(const auto& item:settings) puf_ini_put(&task.ini,item[0],item[1]);
        puf_ini_put(&task.ini,"train.total_timesteps",std::to_string(256*world).c_str());
        puf_ini_put(&task.ini,"train.gpus",argv[3]);
        puf_ini_put(&task.ini,"train.distributed_optimizer",argv[4]);
        const char* bundle=nullptr;
        for(int i=5;i<argc;++i) {
            if(!strncmp(argv[i],"--optimizer=",12)) puf_ini_put(&task.ini,"train.optimizer",argv[i]+12);
            else {
                puf_checkpoint::require(!bundle,"duplicate/unknown test argument"); bundle=argv[i];
            }
        }
#ifdef PUFFER_DECISION_POLICY
        puf_checkpoint::require(bundle!=nullptr,"text test requires bundle");
        puf_ini_put(&task.ini,"policy.bundle",bundle);
        puf_ini_put(&task.ini,"policy.sequence_length","384");
        puf_ini_put(&task.ini,"policy.zero_init_critic","1");
        puf_ini_put(&task.ini,"env.observation_format","1");
#else
        puf_checkpoint::require(bundle==nullptr,"board test does not take a bundle");
#endif
        int status=puf_launch_native_ranks(world,0,0,nullptr,checkpoint_worker,&task);
        puf_ini_free(&task.ini); return status==0 ? 0 : 1;
    } catch(const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return 1; }
}
