#include "include/opt.h"
#include "include/tokenizer.h"

static constexpr auto num_layers = 32;
static constexpr auto num_heads = 32;
static constexpr auto hidden_size = 4096;
static constexpr auto MAX_LEN = 512;

std::optional<OPTImpl> OPTImpl::from_pretrained(
        std::string_view model_path,
        int              dev_id) {
    OPTImpl ctx;

    int status = bm_dev_request(&ctx.handle, dev_id);
    if (status != BM_SUCCESS) return std::nullopt;
    if (!(ctx.bmrt = bmrt_create(ctx.handle))) return std::nullopt;

    printf("Loading model [%s]...\n", model_path.data());

    if (!bmrt_load_bmodel(ctx.bmrt, model_path.data())) return std::nullopt;

    ctx.name_embed = "embedding";
    ctx.name_lm = "lm_head";
    ctx.name_splitkv = "splitkv";

    for (int i = 0; i < num_layers; i++) {
        ctx.name_blocks.push_back("block_" + std::to_string(i));
        ctx.name_blocks_cache.push_back("block_cache_" + std::to_string(i));
    }
    ctx.net_embed = const_cast<bm_net_info_t*>(
            bmrt_get_network_info(ctx.bmrt, ctx.name_embed.c_str()));
    ctx.net_lm = const_cast<bm_net_info_t*>(
            bmrt_get_network_info(ctx.bmrt, ctx.name_lm.c_str()));
    ctx.net_blocks.resize(num_layers);
    ctx.net_blocks_cache.resize(num_layers);

    for (int i = 0; i < num_layers; i++) {
        ctx.net_blocks[i] = const_cast<bm_net_info_t*>(
                bmrt_get_network_info(ctx.bmrt, ctx.name_blocks[i].c_str()));
        ctx.net_blocks_cache[i] =
                const_cast<bm_net_info_t*>(bmrt_get_network_info(
                        ctx.bmrt, ctx.name_blocks_cache[i].c_str()));
    }

    bmrt_tensor(
            &ctx.inputs_embed_512,
            ctx.bmrt,
            ctx.net_embed->input_dtypes[0],
            ctx.net_embed->stages[1].input_shapes[0]);
    bmrt_tensor(
            &ctx.inputs_pid512,
            ctx.bmrt,
            ctx.net_embed->input_dtypes[1],
            ctx.net_embed->stages[1].input_shapes[1]);
    bmrt_tensor(
            &ctx.outputs_embed_512,
            ctx.bmrt,
            ctx.net_embed->output_dtypes[0],
            ctx.net_embed->stages[1].output_shapes[0]);
    bmrt_tensor(
            &ctx.input_token,
            ctx.bmrt,
            ctx.net_embed->input_dtypes[0],
            ctx.net_embed->stages[0].input_shapes[0]);
    bmrt_tensor(
            &ctx.input_pid,
            ctx.bmrt,
            ctx.net_embed->input_dtypes[1],
            ctx.net_embed->stages[0].input_shapes[1]);

    bmrt_tensor(
            &ctx.output_hidden_state,
            ctx.bmrt,
            ctx.net_embed->output_dtypes[0],
            ctx.net_embed->stages[0].output_shapes[0]);

    bmrt_tensor(
            &ctx.current_k,
            ctx.bmrt,
            ctx.net_blocks_cache[0]->output_dtypes[1],
            ctx.net_blocks_cache[0]->stages[0].output_shapes[1]);

    bmrt_tensor(
            &ctx.current_v,
            ctx.bmrt,
            ctx.net_blocks_cache[0]->output_dtypes[2],
            ctx.net_blocks_cache[0]->stages[0].output_shapes[2]);

    bmrt_tensor(
            &ctx.inputs_attention,
            ctx.bmrt,
            ctx.net_blocks[0]->input_dtypes[1],
            ctx.net_blocks[0]->stages[0].input_shapes[1]);

    bmrt_tensor(
            &ctx.next_attention,
            ctx.bmrt,
            ctx.net_blocks_cache[0]->input_dtypes[1],
            ctx.net_blocks_cache[0]->stages[0].input_shapes[1]);

    ctx.past_key.resize(num_layers);
    ctx.past_value.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        bmrt_tensor(
                &ctx.past_key[i],
                ctx.bmrt,
                ctx.net_blocks[0]->output_dtypes[1],
                ctx.net_blocks[0]->stages[0].output_shapes[1]);
        bmrt_tensor(
                &ctx.past_value[i],
                ctx.bmrt,
                ctx.net_blocks[0]->output_dtypes[1],
                ctx.net_blocks[0]->stages[0].output_shapes[2]);
    }

    bmrt_tensor(
            &ctx.inputs_lm,
            ctx.bmrt,
            ctx.net_lm->input_dtypes[0],
            ctx.net_lm->stages[0].input_shapes[0]);

    bmrt_tensor(
            &ctx.outputs_lm,
            ctx.bmrt,
            ctx.net_lm->output_dtypes[0],
            ctx.net_lm->stages[0].output_shapes[0]);

    return ctx;
}

int OPTImpl::forward_first(const std::vector<int>& ids) {
    token_length = ids.size();
    auto attention_mask = std::make_unique<float[]>(MAX_LEN * MAX_LEN);
    auto position_id = std::make_unique<int[]>(MAX_LEN);
    for (int i = 0; i < MAX_LEN; i++) {
        for (int j = i + 1; j < MAX_LEN; j++)
            attention_mask[j + i * MAX_LEN] = -1000.0;
        if (i < token_length) position_id[i] = i;
    }

    bm_memcpy_s2d(handle, inputs_embed_512.device_mem, (void*)ids.data());
    bm_memcpy_s2d(handle, inputs_pid512.device_mem, (void*)position_id.get());
    bm_tensor_t inputs_block[] = {inputs_embed_512, inputs_pid512};
    bm_tensor_t output_block[] = {outputs_embed_512};

    bmrt_launch_tensor_ex(
            bmrt,
            name_embed.c_str(),
            inputs_block,
            2,
            output_block,
            1,
            true,
            false);
    bm_memcpy_s2d(handle, inputs_attention.device_mem, attention_mask.get());
    bm_thread_sync(handle);

    bm_tensor_t hidden_state;
    bmrt_tensor_with_device(
            &hidden_state,
            outputs_embed_512.device_mem,
            outputs_embed_512.dtype,
            net_blocks[0]->stages[0].input_shapes[0]);

    for (int i = 0; i < num_layers; i++) {
        bm_tensor_t inputs_block1[] = {hidden_state, inputs_attention};
        bm_tensor_t outputs_block1[] = {
                hidden_state, past_key[i], past_value[i]};
        bmrt_launch_tensor_ex(
                bmrt,
                name_blocks[i].c_str(),
                inputs_block1,
                2,
                outputs_block1,
                3,
                true,
                false);

        bm_thread_sync(handle);
        move2end(past_key[i]);
        move2end(past_value[i]);
    }

    auto bytes = hidden_state.device_mem.size / MAX_LEN;

    bm_memcpy_d2d(
            handle,
            inputs_lm.device_mem,
            0,
            hidden_state.device_mem,
            (token_length - 1) * bytes,
            bytes);

    bmrt_launch_tensor_ex(
            bmrt, name_lm.c_str(), &inputs_lm, 1, &outputs_lm, 1, true, false);
    bm_thread_sync(handle);

    token_length++;

    int token = 0;
    bm_memcpy_d2s(handle, (void*)&token, outputs_lm.device_mem);
    return token;
}

void OPTImpl::move2end(const bm_tensor_t& cache) {
    auto const total_size = bm_mem_get_device_size(cache.device_mem);
    auto       bytes = total_size / MAX_LEN;
    auto const real_size = token_length * bytes;
    auto const diff_size = (total_size - real_size) / num_heads;
    auto       buffer = std::make_unique<uint8_t[]>(total_size + diff_size);

    bm_memcpy_d2s(handle, buffer.get() + diff_size, cache.device_mem);
    bm_memcpy_s2d(handle, cache.device_mem, buffer.get());
}

int OPTImpl::forward_next() {
    int position_id = token_length;
    bm_memcpy_s2d(handle, input_pid.device_mem, &position_id);

    bm_tensor_t last_token;
    bmrt_tensor_with_device(
            &last_token,
            outputs_lm.device_mem,
            outputs_lm.dtype,
            net_embed->stages[0].input_shapes[0]);

    bm_tensor_t input_tensors[] = {last_token, input_pid};
    bmrt_launch_tensor_ex(
            bmrt,
            name_embed.c_str(),
            input_tensors,
            2,
            &output_hidden_state,
            1,
            true,
            false);

    auto attention_mask = std::make_unique<float[]>(1 + MAX_LEN);
    for (int i = 0; i < MAX_LEN - token_length + 1; i++)
        attention_mask[i] = -1000;

    bm_thread_sync(handle);

    bm_memcpy_s2d(handle, next_attention.device_mem, attention_mask.get());

    bm_tensor_t inputs_embed;
    bmrt_tensor_with_device(
            &inputs_embed,
            output_hidden_state.device_mem,
            output_hidden_state.dtype,
            net_blocks_cache[0]->stages[0].input_shapes[0]);

    for (int i = 0; i < num_layers; i++) {
        bm_tensor_t inputs_block[] = {
                inputs_embed, next_attention, past_key[i], past_value[i]};
        bm_tensor_t output_block[] = {inputs_embed, current_k, current_v};

        bmrt_launch_tensor_ex(
                bmrt,
                name_blocks_cache[i].c_str(),
                inputs_block,
                4,
                output_block,
                3,
                true,
                false);
        bm_tensor_t block1[] = {current_k, current_v};
        bm_tensor_t block2[] = {past_key[i], past_value[i]};

        bmrt_launch_tensor_ex(
                bmrt, name_splitkv.c_str(), block1, 2, block2, 2, true, false);
        bm_thread_sync(handle);
    }

    bm_tensor_t lm_input;
    bmrt_tensor_with_device(
            &lm_input,
            output_hidden_state.device_mem,
            output_hidden_state.dtype,
            net_lm->stages[0].input_shapes[0]);

    bm_tensor_t input[] = {lm_input};
    bm_tensor_t output[] = {outputs_lm};

    bmrt_launch_tensor_ex(
            bmrt, name_lm.c_str(), input, 1, output, 1, true, false);
    token_length++;
    int token = 0;
    bm_memcpy_d2s(handle, &token, outputs_lm.device_mem);

    bm_thread_sync(handle);

    return token;
}

std::optional<OPTModel> OPTModel::from_pretrained(
        std::string_view model_path,
        int              dev_id) {
    auto tokenizer = GPT2Tokenizer::from_pretrained(model_path);
    if (!tokenizer) return std::nullopt;
    auto ctx = OPTImpl::from_pretrained(model_path, dev_id);
    if (!ctx) return std::nullopt;

    OPTModel model;
    model.tokenizer = tokenizer.value();
    model.impl = ctx.value();

    return model;
}

std::vector<int> OPTModel::encode(std::string_view input) {
    auto             ids = tokenizer.encode(input);
    std::vector<int> result(1 + ids.size());
    result[0] = 2;
    memcpy(&result[1], ids.data(), sizeof(int) * ids.size());
    return result;
}

void OPTModel::stream_generate(
        const std::vector<int>& ids,
        int                     max_new_length) {
    int cnt = 1;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto token = impl.forward_first(ids);
    while (++cnt < max_new_length) {
        std::cout << tokenizer.decode(std::vector<int>{token}, true)
                  << std::flush;
        token = impl.forward_next();
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto time_count = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_time - start_time)
                              .count();
    std::cout << "\n\nTime: " << time_count << " ms Token: " << cnt
              << " Ratio: " << cnt * 1000.0 / time_count << " Tokens/sec\n";
}
