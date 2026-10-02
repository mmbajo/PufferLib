// Full Transformer policy through Puffer's existing encoder/network/decoder
// callbacks. The model computes both heads; the latter two stages pass through.
#ifndef PRECISION_FLOAT
#error "decision_snake Transformer currently requires --float"
#endif
#include "../../src/decision_transformer.cuh"

struct DecisionSnakeWeights {
    decision::Config config;
    std::vector<Prec> parameters;
    std::vector<size_t> counts;
};
struct DecisionSnakeWorkspace {
    decision::Model* model;
    std::vector<Prec> gradients;
    bool bound=false;
};
struct DecisionSnakeActivations {
    DecisionSnakeWorkspace* workspace;
    Int boards;
    Prec output;
    Float grad_logits, grad_values;
};

static __global__ void decision_snake_decode_board(int* output,const float* input,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=(int)input[i]-1;
}
static __global__ void decision_snake_pack(float* out,const float* logits,const float* values,int batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<batch*5)out[i]=(i%5==4)?values[i/5]:logits[(i/5)*4+i%5];
}
static __global__ void decision_snake_unpack(const float* input,float* logits,float* values,int batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<batch*5) {
        if(i%5==4)values[i/5]=input[i];
        else logits[(i/5)*4+i%5]=input[i];
    }
}

static void* decision_snake_weights(void* self) {
    Encoder* encoder=(Encoder*)self;
    auto* weights=new DecisionSnakeWeights;
    weights->config.width=encoder->out_dim;
    weights->config.layers=encoder->num_layers;
    weights->config.head_pooling=true;
    weights->config.relative_coordinates=true;
    decision::Model layout(weights->config,1);
    weights->parameters.resize(layout.parameters().size());
    for(size_t i=0;i<layout.parameters().size();++i) {
        const auto& p=layout.parameters()[i];
        Prec tensor={};
        for(size_t j=0;j<p.shape.size();++j)tensor.shape[j]=p.shape[j];
        // Puffer's flat parameter/gradient ABI assumes no alignment gaps.
        // Only the scalar value bias needs padding in this FP32 architecture.
        if(p.count%4) {
            assert(p.shape.size()==1);
            tensor.shape[0]=(p.count+3)&~size_t(3);
        }
        weights->parameters[i]=tensor; weights->counts.push_back(p.count);
    }
    return weights;
}
static void decision_snake_register_parameters(void* opaque,Allocator* alloc) {
    auto* w=(DecisionSnakeWeights*)opaque;
    for(auto& parameter:w->parameters)alloc_register(alloc,&parameter);
}
static void decision_snake_initialize(void* opaque,ulong* seed,cudaStream_t stream) {
    auto* w=(DecisionSnakeWeights*)opaque;
    decision::Model initial(w->config,1,(*seed)++,stream);
    for(size_t i=0;i<w->parameters.size();++i) {
        // Zero the alignment padding as well as initializing actual weights.
        cudaMemsetAsync(w->parameters[i].data,0,numel(w->parameters[i].shape)*sizeof(float),stream);
        cudaMemcpyAsync(w->parameters[i].data,initial.parameters()[i].data,
            w->counts[i]*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    }
    cudaStreamSynchronize(stream);
}
static void decision_snake_register_common(void* opaque,void* activations,
        Allocator* acts,Allocator* grads,int batch) {
    auto* w=(DecisionSnakeWeights*)opaque;
    auto* a=(DecisionSnakeActivations*)activations;
    a->workspace=new DecisionSnakeWorkspace;
    a->workspace->model=new decision::Model(w->config,batch);
    a->boards={.shape={batch,100}}; alloc_register(acts,&a->boards);
    a->output={.shape={batch,5}}; alloc_register(acts,&a->output);
    if(grads) {
        a->workspace->gradients=w->parameters;
        for(auto& gradient:a->workspace->gradients) {
            gradient.data=nullptr; alloc_register(grads,&gradient);
        }
        a->grad_logits={.shape={batch,4}}; alloc_register(acts,&a->grad_logits);
        a->grad_values={.shape={batch}}; alloc_register(acts,&a->grad_values);
    }
}
static void decision_snake_register_rollout(void* w,void* a,Allocator* acts,int batch) {
    decision_snake_register_common(w,a,acts,nullptr,batch);
}
static void decision_snake_register_train(void* w,void* a,Allocator* acts,Allocator* grads,int batch) {
    decision_snake_register_common(w,a,acts,grads,batch);
}
static Prec decision_snake_forward(void* opaque,void* activations,Prec input,cudaStream_t stream) {
    auto* w=(DecisionSnakeWeights*)opaque;
    auto* a=(DecisionSnakeActivations*)activations;
    auto* model=a->workspace->model;
    if(!a->workspace->bound) {
        std::vector<float*> data,gradients;
        for(size_t i=0;i<w->parameters.size();++i) {
            data.push_back(w->parameters[i].data);
            gradients.push_back(a->workspace->gradients.empty()?model->parameters()[i].grad:
                a->workspace->gradients[i].data);
        }
        model->bind_parameters(data,gradients); a->workspace->bound=true;
    }
    model->set_stream(stream);
    int batch=(int)(numel(input.shape)/100);
    decision_snake_decode_board<<<grid_size(batch*100),BLOCK_SIZE,0,stream>>>(
        a->boards.data,input.data,batch*100);
    auto output=model->forward(a->boards.data,batch);
    decision_snake_pack<<<grid_size(batch*5),BLOCK_SIZE,0,stream>>>(
        a->output.data,output.logits,output.values,batch);
    return Prec{.data=a->output.data,.shape={batch,5}};
}
static void decision_snake_backward(void*,void* activations,Prec gradient,cudaStream_t stream) {
    auto* a=(DecisionSnakeActivations*)activations;
    int batch=(int)(numel(gradient.shape)/5);
    decision_snake_unpack<<<grid_size(batch*5),BLOCK_SIZE,0,stream>>>(
        gradient.data,a->grad_logits.data,a->grad_values.data,batch);
    a->workspace->model->set_stream(stream);
    // Stock Puffer backward kernels overwrite gradients. This model accumulates
    // derivatives, so clear its registered slots before each minibatch backward.
    a->workspace->model->zero_grad();
    a->workspace->model->backward(a->grad_logits.data,a->grad_values.data);
}
static void create_decision_snake_encoder(Encoder* encoder) {
    encoder->create_weights=decision_snake_weights;
    encoder->reg_params=decision_snake_register_parameters;
    encoder->reg_train=decision_snake_register_train;
    encoder->reg_rollout=decision_snake_register_rollout;
    encoder->init_weights=decision_snake_initialize;
    encoder->forward=decision_snake_forward;
    encoder->backward=decision_snake_backward;
    encoder->activation_size=sizeof(DecisionSnakeActivations);
}

static void decision_snake_no_params(void*,Allocator*) {}
static void decision_snake_no_init(void*,ulong*,cudaStream_t) {}
static void decision_snake_no_rollout(void*,void*,Allocator*,int) {}
static void decision_snake_no_train(void*,void*,Allocator*,Allocator*,int) {}
static void* decision_snake_identity_weights(void*) {return calloc(1,1);}
static Prec decision_snake_identity_forward(void*,Prec input,Prec,void*,cudaStream_t) {return input;}
static Prec decision_snake_identity_train(void*,Prec input,Prec,Prec,void*,int,cudaStream_t) {return input;}
static Prec decision_snake_identity_backward(void*,Prec gradient,void*,cudaStream_t) {return gradient;}
static void create_decision_snake_network(Network* network) {
    network->forward=decision_snake_identity_forward;
    network->forward_train=decision_snake_identity_train;
    network->backward=decision_snake_identity_backward;
    network->create_weights=decision_snake_identity_weights;
    network->reg_params=decision_snake_no_params;
    network->reg_train=decision_snake_no_train;
    network->reg_rollout=decision_snake_no_rollout;
    network->init_weights=decision_snake_no_init;
}

// Retain the stock DecoderWeights layout: the native runner accesses continuous
// and logstd through it even when a custom decoder's callbacks are installed.
static void* decision_snake_decoder_weights(void*) {
    auto* w=(DecoderWeights*)calloc(1,sizeof(DecoderWeights));
    w->hidden_dim=5; w->output_dim=4; w->continuous=false; return w;
}
static Prec decision_snake_decoder_forward(void*,void*,Prec input,cudaStream_t) {return input;}
static void decision_snake_decoder_train(void*,void* activations,Allocator* acts,Allocator*,int batch) {
    auto* a=(DecoderActivations*)activations;
    a->grad_out={.shape={batch,5}}; alloc_register(acts,&a->grad_out);
}
static Prec decision_snake_decoder_backward(void*,void* activations,Float logits,Float,Float values,cudaStream_t stream) {
    auto* a=(DecoderActivations*)activations;
    int batch=(int)(numel(logits.shape)/4);
    assemble_decoder_grad<<<grid_size(batch*5),BLOCK_SIZE,0,stream>>>(
        a->grad_out.data,logits.data,values.data,batch,4,5);
    return Prec{.data=a->grad_out.data,.shape={batch,5}};
}
static void create_decision_snake_decoder(Decoder* decoder) {
    decoder->forward=decision_snake_decoder_forward;
    decoder->backward=decision_snake_decoder_backward;
    decoder->create_weights=decision_snake_decoder_weights;
    decoder->reg_params=decision_snake_no_params;
    decoder->reg_train=decision_snake_decoder_train;
    decoder->reg_rollout=decision_snake_no_rollout;
    decoder->init_weights=decision_snake_no_init;
}
