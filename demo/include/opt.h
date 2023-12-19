#ifndef DEMO_H
#define DEMO_H

#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include "tokenizer.h"

struct OPTImpl {
    bm_handle_t                   handle;
    void*                         bmrt;
    std::vector<bm_net_info_t*>   net_blocks;
    std::vector<std::string>      name_blocks;
    std::vector<bm_net_info_t*>   net_blocks_cache;
    std::vector<std::string>      name_blocks_cache;
    bm_net_info_t*                net_embed;
    std::string                   name_embed;
    bm_net_info_t*                net_lm;
    std::string                   name_lm;
    bm_net_info_t*                net_splitkv;
    std::string                   name_splitkv;
    bm_tensor_t                   inputs_embed_512, outputs_embed_512;
    bm_tensor_t                   input_token, input_pid, inputs_pid512;
    bm_tensor_t                   output_hidden_state;
    bm_tensor_t                   current_k, current_v;
    bm_tensor_t                   inputs_attention, next_attention;
    std::vector<bm_tensor_t>      past_key, past_value;
    bm_tensor_t                   inputs_lm, outputs_lm;
    int                           token_length;
    static std::optional<OPTImpl> from_pretrained(std::string_view, int);

    void init();
    void move2end(const bm_tensor_t& cache);
    int  forward_first(const std::vector<int>& token_ids);
    int  forward_next();
};

struct OPTModel {
    GPT2Tokenizer tokenizer;
    OPTImpl       impl;

    static std::optional<OPTModel> from_pretrained(std::string_view, int);
    std::vector<int>               encode(std::string_view);
    void stream_generate(const std::vector<int>&, int);
};

#endif