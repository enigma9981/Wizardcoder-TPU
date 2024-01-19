#include "include/wizardcoder.h"
#include <cnpy.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include "bmdef.h"
#include "bmlib_runtime.h"
#include "bmruntime_interface.h"
#include "include/config.h"
#include "include/tokenizer.h"
#include "include/utils.h"

static constexpr int MAX_LEN = 512;

void compare_vectors(
        const float* a,
        const float* b,
        int          n,
        int          max_idx,
        int          topk = 20) {
    float v1 = 0, v2 = 0, v3 = 0, d = 0;

    using PFI = std::pair<float, int>;
    std::vector<PFI> rela;

    for (int i = 0; i < max_idx; i++)
        v1 += a[i] * a[i], v2 += b[i] * b[i], v3 += a[i] * b[i],
                d += (a[i] - b[i]) * (a[i] - b[i]),
                rela.push_back({std::abs((a[i] - b[i]) / b[i]), i});

    std::cout << rela.size() << '\n';

    std::cout << "\n\nSimi:" << std::sqrt(v3 * v3 / v1 / v2) << '\n';
    std::cout << "\n\nEulide:" << std::sqrt(d) << '\n';

    std::cout << "INT8: ";
    for (int i = 0; i < topk; i++) {
        std::cout << a[i] << ' ';
    }
    std::cout << "\n\n\n";

    std::cout << "ONNX: ";
    for (int i = 0; i < topk; i++) {
        std::cout << b[i] << ' ';
    }
    std::cout << '\n';

    std::sort(rela.begin(), rela.end(), [](PFI a, PFI b) {
        return a.first > b.first;
    });

    std::cout << '\n';
}

template <typename T>
void dump_tensor(bm_handle_t& handle, const bm_tensor_t& t) {
    std::cout << "Shape: \n";
    for (int i = 0; i < t.shape.num_dims; i++) {
        std::cout << t.shape.dims[i] << ", ";
    }
    std::cout << '\n';
    int  cnt = bm_mem_get_device_size(t.device_mem) / sizeof(T);
    auto buffer = std::make_unique<T[]>(cnt);
    std::cout << "cnt: " << cnt << '\n';
    bm_memcpy_d2s(handle, buffer.get(), t.device_mem);

    int cc = 0;
    for (int i = 0; i < cnt; i++) {
        if (std::isnan(buffer[i])) ++cc;
    }
    std::cout << "CC: " << cc << '\n';
}

std::optional<WizardCoderImpl> WizardCoderImpl::from_pretrained(
        std::string_view        model_path,
        const std::vector<int>& devids) {
    WizardCoderImpl ctx;

    auto const cfg = AutoConfig::from_pretrained(model_path);
    if (cfg == std::nullopt)
        return std::cerr << "No config found\n", std::nullopt;

    int num_device = devids.size();
    ctx.num_device = num_device;
    ctx.blocks.resize(cfg->num_layers);
    ctx.blocks_cache.resize(cfg->num_layers);

    for (auto&& block : ctx.blocks) {
        block.attention_mask.resize(num_device);
        block.hidden_states.resize(num_device);
        block.input_states.resize(num_device);
        block.past_layers.resize(num_device);
    }

    for (auto&& block_cache : ctx.blocks_cache) {
        block_cache.current_cache.resize(num_device);
        block_cache.past_cache.resize(num_device);
        block_cache.attention_mask.resize(num_device);
        block_cache.hidden_states.resize(num_device);
        block_cache.input_states.resize(num_device);
    }

    for (auto id : devids) {
        bm_handle_t handle;
        if (bm_dev_request(&handle, id) != BM_SUCCESS) return std::nullopt;
        ctx.handles.push_back(handle);
    }

    ctx.handle = ctx.handles[0];
    auto& handle = ctx.handles[0];

    if (!(ctx.bmrt = bmrt_create_ex(&handle, num_device))) return std::nullopt;
    auto& bmrt = ctx.bmrt;
    if (!bmrt_load_bmodel(bmrt, model_path.data())) return std::nullopt;

    const char** network_names{nullptr};
    bmrt_get_network_names(bmrt, &network_names);
    int num = bmrt_get_network_number(bmrt);
    for (int i = 0; i < num; i++) {
        ctx.networks[network_names[i]] =
                bmrt_get_network_info(bmrt, network_names[i]);
    }
    auto& networks = ctx.networks;

    [&]() {
        bmrt_tensor(
                &ctx.embedding.input_ids_512,
                bmrt,
                networks["embedding"]->input_dtypes[0],
                networks["embedding"]->stages[1].input_shapes[0]);
        bmrt_tensor(
                &ctx.embedding.input_pos_512,
                bmrt,
                networks["embedding"]->input_dtypes[1],
                networks["embedding"]->stages[1].input_shapes[1]);
        bmrt_tensor(
                &ctx.embedding.hidden_states_512,
                bmrt,
                networks["embedding"]->output_dtypes[0],
                networks["embedding"]->stages[1].output_shapes[0]);
        bmrt_tensor(
                &ctx.embedding.input_ids_1,
                bmrt,
                networks["embedding"]->input_dtypes[0],
                networks["embedding"]->stages[0].input_shapes[0]);
        bmrt_tensor(
                &ctx.embedding.input_pos_1,
                bmrt,
                networks["embedding"]->input_dtypes[1],
                networks["embedding"]->stages[0].input_shapes[1]);
        bmrt_tensor(
                &ctx.embedding.hidden_states_1,
                bmrt,
                networks["embedding"]->output_dtypes[0],
                networks["embedding"]->stages[0].output_shapes[0]);
    }();

    [&]() {
        for (int i = 0; i < cfg->num_layers; i++) {
            auto  name = std::string{"block_"} + std::to_string(i);
            auto  block_net = bmrt_get_network_info(bmrt, name.c_str());
            int   in_num = block_net->input_num / num_device;
            int   out_num = block_net->output_num / num_device;
            auto& block = ctx.blocks[i];

            for (int j = 0; j < num_device; j++) {
                bmrt_tensor_ex(
                        &block.input_states[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 0],
                        block_net->input_dtypes[j * in_num + 0],
                        block_net->stages[0].input_shapes[j * in_num + 0]);
                bmrt_tensor_ex(
                        &block.attention_mask[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 1],
                        block_net->input_dtypes[j * in_num + 1],
                        block_net->stages[0].input_shapes[j * in_num + 1]);
                bmrt_tensor_ex(
                        &block.hidden_states[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 0],
                        block_net->output_dtypes[j * out_num + 0],
                        block_net->stages[0].output_shapes[j * out_num + 0]);
                bmrt_tensor_ex(
                        &block.past_layers[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 1],
                        block_net->output_dtypes[j * out_num + 1],
                        block_net->stages[0].output_shapes[j * out_num + 1]);
            }
        }
    }();

    [&]() {
        for (int i = 0; i < cfg->num_layers; i++) {
            auto name = std::string{"block_cache_"} + std::to_string(i);
            auto block_net = bmrt_get_network_info(bmrt, name.c_str());
            // auto devs = block->input_loc_devices;
            int   in_num = block_net->input_num / num_device;
            int   out_num = block_net->output_num / num_device;
            auto& block = ctx.blocks_cache[i];
            for (int j = 0; j < num_device; j++) {
                bmrt_tensor_ex(
                        &block.input_states[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 0],
                        block_net->input_dtypes[j * in_num + 0],
                        block_net->stages[0].input_shapes[j * in_num + 0]);

                bmrt_tensor_ex(
                        &block.past_cache[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 1],
                        block_net->input_dtypes[j * in_num + 1],
                        block_net->stages[0].input_shapes[j * in_num + 1]);

                bmrt_tensor_ex(
                        &block.attention_mask[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 2],
                        block_net->input_dtypes[j * in_num + 2],
                        block_net->stages[0].input_shapes[j * in_num + 2]);

                bmrt_tensor_ex(
                        &block.hidden_states[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 0],
                        block_net->output_dtypes[j * out_num + 0],
                        block_net->stages[0].output_shapes[j * out_num + 0]);

                bmrt_tensor_ex(
                        &block.current_cache[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 1],
                        block_net->output_dtypes[j * out_num + 1],
                        block_net->stages[0].output_shapes[j * out_num + 1]);
            }
        }
    }();

    [&]() {
        auto lm_head = bmrt_get_network_info(bmrt, "lm_head");
        bmrt_tensor(
                &ctx.lm_head.hidden_states,
                bmrt,
                lm_head->input_dtypes[0],
                lm_head->stages[0].input_shapes[0]);
        bmrt_tensor(
                &ctx.lm_head.token,
                bmrt,
                lm_head->output_dtypes[0],
                lm_head->stages[0].output_shapes[0]);
    }();

    ctx.model_config = cfg.value();

    std::cout << FBLU("Loaded\n");
    return ctx;
}

int WizardCoderImpl::forward_first(const std::vector<int>& token_ids) {
    token_length = token_ids.size();
    auto attention_mask = std::make_unique<float[]>(MAX_LEN * MAX_LEN);
    auto position_id = std::make_unique<int[]>(MAX_LEN);
    for (int i = 0; i < MAX_LEN; i++) {
        for (int j = i + 1; j < MAX_LEN; j++)
            attention_mask[j + i * MAX_LEN] = -1000.0;
        if (i < token_length) position_id[i] = i;
    }

    std::vector<int>   one_input_nums{1};
    std::vector<int>   num_device_inputs_nums(num_device, 1);
    std::vector<void*> pos_id_data{position_id.get()};
    std::vector<void*> tok_id_data{(void*)token_ids.data()};

    std::vector<void*> attention_mask_data(
            num_device, (void*)attention_mask.get());

    bmrt_memcpy_s2d_parallel(
            bmrt,
            &embedding.input_ids_512,
            tok_id_data.data(),
            one_input_nums.data(),
            1);
    bmrt_memcpy_s2d_parallel(
            bmrt,
            &embedding.input_pos_512,
            pos_id_data.data(),
            one_input_nums.data(),
            1);

    bmrt_memcpy_s2d_parallel(
            bmrt,
            blocks.begin()->attention_mask.data(),
            attention_mask_data.data(),
            num_device_inputs_nums.data(),
            num_device);

    bm_tensor_t input_blocks[] = {
            embedding.input_ids_512, embedding.input_pos_512};

    bmrt_launch_tensor_ex(
            bmrt,
            "embedding",
            input_blocks,
            2,
            &embedding.hidden_states_512,
            1,
            true,
            false);

    bm_thread_sync(handle);

    std::vector<bm_tensor_t> inputs_block;
    std::vector<bm_tensor_t> outputs_block;

    for (int i = 0; i < num_device; i++) {
        embedding.hidden_states_512.shape = blocks[0].input_states[0].shape;
        inputs_block.push_back(embedding.hidden_states_512);
        inputs_block.push_back(blocks[0].attention_mask[i]);
        outputs_block.push_back(embedding.hidden_states_512);
        outputs_block.push_back(blocks[0].past_layers[i]);
    }

    for (int i = 0; i < model_config.num_layers; i++) {
        auto name = std::string{"block_"} + std::to_string(i);
        for (int j = 0; j < num_device; j++) {
            outputs_block[1] = blocks[i].past_layers[j];
        }

        bmrt_launch_tensor_ex(
                bmrt,
                name.c_str(),
                inputs_block.data(),
                inputs_block.size(),
                outputs_block.data(),
                outputs_block.size(),
                true,
                false);

        for (int j = 0; j < num_device; j++) {
            move2end(blocks[i].past_layers[j]);
        }

        bm_thread_sync(handle);
    }

    auto bytes =
            bm_mem_get_device_size(embedding.hidden_states_512.device_mem) /
            MAX_LEN;

    bm_memcpy_d2d(
            handle,
            lm_head.hidden_states.device_mem,
            0,
            embedding.hidden_states_512.device_mem,
            (token_length - 1) * bytes,
            bytes);

    bmrt_launch_tensor_ex(
            bmrt,
            "lm_head",
            &lm_head.hidden_states,
            1,
            &lm_head.token,
            1,
            true,
            false);
    int token = 0;
    bm_memcpy_d2s(handle, &token, lm_head.token.device_mem);

    ++token_length;
    return token;
}

void WizardCoderImpl::move2end(const bm_tensor_t& cache) {
    auto sz = bm_mem_get_device_size(cache.device_mem);
    auto bytes = sz / MAX_LEN;
    auto x = model_config.hidden_size / model_config.num_heads * 2;
    auto len = token_length * bytes;
    bm_memcpy_d2d(handle, cache.device_mem, sz - len, cache.device_mem, 0, len);
}

int WizardCoderImpl::forward_next() {
    int                pid = token_length - 1;
    std::vector<void*> input_pid_data{&pid};
    std::vector<int>   embedding_inputs_num{1};
    bmrt_memcpy_s2d_parallel(
            bmrt,
            &embedding.input_pos_1,
            input_pid_data.data(),
            embedding_inputs_num.data(),
            1);

    bmrt_tensor_with_device(
            &embedding.input_ids_1,
            lm_head.token.device_mem,
            embedding.input_ids_1.dtype,
            embedding.input_ids_1.shape);

    bm_tensor_t input_blocks[] = {embedding.input_ids_1, embedding.input_pos_1};
    bmrt_launch_tensor_ex(
            bmrt,
            "embedding",
            input_blocks,
            2,
            &embedding.hidden_states_1,
            1,
            true,
            false);

    bm_thread_sync(handle);

    auto attention_mask = std::make_unique<float[]>(1 + MAX_LEN);
    for (int i = 0; i < MAX_LEN - token_length + 1; i++)
        attention_mask[i] = -1000;

    std::vector<int>   input_nums(num_device, 1);
    std::vector<void*> attention_mask_data(num_device, attention_mask.get());

    bmrt_memcpy_s2d_parallel(
            bmrt,
            blocks_cache.begin()->attention_mask.data(),
            attention_mask_data.data(),
            input_nums.data(),
            num_device);

    std::vector<bm_tensor_t> inputs_block;
    std::vector<bm_tensor_t> outputs_block;

    for (int i = 0; i < num_device; i++) {
        inputs_block.push_back(embedding.hidden_states_1);
        inputs_block.push_back(blocks[0].past_layers[i]);
        inputs_block.push_back(blocks_cache[0].attention_mask[i]);
        outputs_block.push_back(embedding.hidden_states_1);
        outputs_block.push_back(blocks_cache[0].current_cache[i]);
    }

    for (int i = 0; i < model_config.num_layers; i++) {
        auto name = std::string{"block_cache_"} + std::to_string(i);

        for (int j = 0; j < num_device; j++) {
            inputs_block[1] = blocks[i].past_layers[j];
            outputs_block[1] = blocks_cache[i].current_cache[j];
        }

        bmrt_launch_tensor_ex(
                bmrt,
                name.c_str(),
                inputs_block.data(),
                inputs_block.size(),
                outputs_block.data(),
                outputs_block.size(),
                true,
                false);

        bm_thread_sync(handle);

        auto totalsize = bm_mem_get_device_size(
                                 blocks_cache[0].current_cache[0].device_mem) /
                513;
        for (int j = 0; j < num_device; j++) {
            bm_memcpy_d2d(
                    handle,
                    blocks[i].past_layers[j].device_mem,
                    0,
                    blocks_cache[i].current_cache[j].device_mem,
                    totalsize,
                    totalsize * 512);
        }
    }

    bmrt_launch_tensor_ex(
            bmrt,
            "lm_head",
            &embedding.hidden_states_1,
            1,
            &lm_head.token,
            1,
            true,
            false);
    bm_thread_sync(handle);
    int token = 0;
    ++token_length;
    bm_memcpy_d2s(handle, &token, lm_head.token.device_mem);

    return token;
}

std::vector<int> WizardCoderModel::encode(std::string_view input_str) {
    return tokenizer.encode(input_str);
}

void WizardCoderModel::init(
        std::string_view        model_path,
        const std::vector<int>& dev_ids) {
    auto ctx = WizardCoderImpl::from_pretrained(model_path, dev_ids);
    if (!ctx) std::cerr << FRED("Ctx Error\n");
    inner = std::move(ctx.value());
    tokenizer = std::move(GPT2Tokenizer::from_pretrained(model_path).value());
}

std::optional<WizardCoderModel> WizardCoderModel::from_pretrained(
        std::string_view        model_path,
        const std::vector<int>& dev_ids) {
    WizardCoderModel model;
    auto inner = WizardCoderImpl::from_pretrained(model_path, dev_ids);
    if (inner == std::nullopt) return std::nullopt;
    model.inner = std::move(inner.value());

    auto tokenizer = GPT2Tokenizer::from_pretrained(model_path);
    if (tokenizer == std::nullopt) return std::nullopt;

    model.tokenizer = std::move(tokenizer.value());

    return model;
}

std::string WizardCoderModel::build_prompt(std::string_view input_str) const {
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
           "### Instruction:\n" +
            std::string{input_str} + "\n\n### Response:";
}

void WizardCoderModel::stream_generate(
        const std::vector<int>& input_ids,
        int                     max_new_length) {
    int cnt = 1;

    auto const input_token_len = input_ids.size();
    auto       start_time = std::chrono::high_resolution_clock::now();
    auto       token = inner.forward_first(input_ids);

    auto FTL = get_elasped(start_time);

    start_time = std::chrono::high_resolution_clock::now();

    while (++cnt < max_new_length && cnt + input_token_len <= MAX_LEN) {
        auto result = tokenizer.decode_id(token, true);
        if (result == "<|endoftext|>") break;
        std::cout << result << std::flush;
        token = inner.forward_next();
    }

    auto total = get_elasped(start_time);

    std::cout << FYEL("\n\nInference Time: ") << (total + FTL)
              << FYEL(" ms\nToken: ") << cnt << FYEL(" FTL: ") << FTL
              << FYEL(" ms\nRate: ") << (cnt - 1) * 1000.0 / total
              << FYEL(" Token/Sec\n");
}
