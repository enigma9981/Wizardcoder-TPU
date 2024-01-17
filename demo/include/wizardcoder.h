#ifndef WIZARDCODER_H
#define WIZARDCODER_H

#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "bmdef.h"
#include "tokenizer.h"

/*
The legacy WizardCoder runtime on BM1684X, including WizardCoder-15B-V1.0 and
WizardCoder-1B-V1.0. The new version of WizardCoder is based on CodeLLAMA and
may not be compatible with this runtime.
*/

struct WizardCoderImpl {
    WizardCoderImpl() {}

    WizardCoderImpl(const WizardCoderImpl&) = delete;
    WizardCoderImpl& operator=(const WizardCoderImpl&) = delete;
    WizardCoderImpl(WizardCoderImpl&&) noexcept = default;
    WizardCoderImpl& operator=(WizardCoderImpl&&) noexcept = default;

    std::vector<bm_handle_t> handles;
    std::vector<int>         dev_ids;
    int                      num_device;
    bm_handle_t              handle;
    void*                    bmrt;

    struct WizardCoderEmbedding {
        bm_tensor_t input_ids_512, input_pos_512;
        bm_tensor_t input_ids_1, input_pos_1;
        bm_tensor_t hidden_states_512, hidden_states_1;
    } embedding;

    struct WizardCoderBlock {
        std::vector<bm_tensor_t> input_states;
        std::vector<bm_tensor_t> attention_mask;
        std::vector<bm_tensor_t> hidden_states;
        std::vector<bm_tensor_t> past_layers;
    };

    struct WizardCoderBlockCache {
        std::vector<bm_tensor_t> input_states;
        std::vector<bm_tensor_t> past_cache;
        std::vector<bm_tensor_t> attention_mask;
        std::vector<bm_tensor_t> hidden_states;
        std::vector<bm_tensor_t> current_cache;
    };

    std::vector<WizardCoderBlock>      blocks;
    std::vector<WizardCoderBlockCache> blocks_cache;

    struct WizardCoderLmHead {
        bm_tensor_t hidden_states;
        bm_tensor_t token;
    } lm_head;

    int token_length;

    std::unordered_map<std::string_view, const bm_net_info_t*> networks;

    static std::optional<WizardCoderImpl> from_pretrained(
            std::string_view,
            const std::vector<int>& devids);

    void move2end(const bm_tensor_t& cache);
    int  forward_first(const std::vector<int>& token_ids);
    int  forward_next();
    void deinit();
};

struct WizardCoderModel {
    WizardCoderImpl inner;
    GPT2Tokenizer   tokenizer;

    static std::optional<WizardCoderModel> from_pretrained(
            std::string_view,
            const std::vector<int>&);

    std::vector<int> encode(std::string_view);
    void stream_generate(const std::vector<int>& input_ids, int max_new_length);
    std::string build_prompt(std::string_view) const;
};

inline long long get_elasped(
        std::chrono::time_point<
                std::chrono::system_clock,
                std::chrono::duration<long, std::ratio<1, 1000000000>>>& last) {
    auto now = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(now - last)
            .count();
}

#endif