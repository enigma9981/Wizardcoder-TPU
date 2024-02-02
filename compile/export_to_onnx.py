import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description='export Wizardcoder onnx.')
parser.add_argument('--model_path', type=str,
                    default="../Wizardcoder-15B-V1.0", help='path to the torch model.')
parser.add_argument('--max_length', type=int, default=512,
                    help="max sequence length")

args = parser.parse_args()
model_path = args.model_path
MAX_LEN = args.max_length
onnx_model_path = f'./tmp'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
omodel = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True).eval()


for param in omodel.parameters():
    param.requires_grad = False

config = omodel.config

transformer = omodel.transformer


num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
head_dim = hidden_size // num_attention_heads


print(f'Layers: {num_layers}\nHidden size: {hidden_size}\n')


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = transformer.wte
        self.wpe = transformer.wpe

    def forward(self, input_ids, position_ids):
        return self.wte(input_ids) + self.wpe(position_ids)


class Block(nn.Module):
    def __init__(self, idx) -> None:
        super().__init__()
        self.layer = transformer.h[idx]

    def forward(self, hidden_states, attention_mask=None):
        hidden_states, past_layer = self.layer(hidden_states, use_cache=True,
                                               attention_mask=attention_mask)
        return hidden_states, past_layer


class BlockCache(nn.Module):
    def __init__(self, idx) -> None:
        super().__init__()
        self.layer = transformer.h[idx]

    def forward(self, hidden_states, layer_past, attention_mask=None):
        hidden_states, past_layer = self.layer(hidden_states, layer_past,
                                               use_cache=True, attention_mask=attention_mask)
        return hidden_states, past_layer


class LmHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = transformer.ln_f
        self.lm_head = omodel.lm_head

    def forward(self, hidden_states):
        x = self.ln(hidden_states)
        logits = self.lm_head(x)
        _, token = torch.topk(logits, 1)
        return token


def get_token(lmhead, hidden_states):
    token = lmhead(hidden_states).view(-1)

    print(f'{tokenizer.decode(token.tolist())}', end="", flush=True)
    return token


def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def net_test():
    embeds = Embedding()
    blocks = [Block(i) for i in range(num_layers)]
    blocks_cache = [BlockCache(i) for i in range(num_layers)]
    lm_head = LmHead()
    init_ids = tokenizer.encode(generate_prompt(
        'Write a Python code to count 1 to 10.'))
    token_len = len(init_ids)
    init_ids = torch.tensor(init_ids)
    init_pos = torch.arange(token_len)
    print(token_len)
    hidden_states = embeds(init_ids, init_pos).view(1, -1, hidden_size)

    past_layers = []
    for i in range(num_layers):
        hidden_states, kv_cache = blocks[i](hidden_states)
        past_layers.append(kv_cache)

    token = get_token(lm_head, hidden_states[:, token_len - 1:, :])

    ng = 1
    token_len += 1

    while ng < 50:
        hidden_states = embeds(token, torch.tensor(
            token_len - 1)).view(1, -1, hidden_size)
        for i in range(num_layers):
            hidden_states, current_cache = blocks_cache[i](
                hidden_states, past_layers[i])
            past_layers[i] = current_cache
        token = get_token(lm_head, hidden_states)
        token_len += 1
        ng += 1


compare_data_dir = './tmp/compare_token_4'


def net_test_fixed_length(prompt):
    embeds = Embedding()
    blocks = [Block(i) for i in range(num_layers)]
    blocks_cache = [BlockCache(i) for i in range(num_layers)]
    lm_head = LmHead()
    init_ids = tokenizer.encode(generate_prompt(
        prompt))

    print(init_ids)
    token_len = len(init_ids)

    init_ids = init_ids + (MAX_LEN - token_len) * [0]
    init_ids = torch.tensor(init_ids)

    init_pos = list(range(token_len)) + (MAX_LEN - token_len) * [0]
    init_pos = torch.tensor(init_pos)

    attention_mask = -1000 * \
        torch.ones((MAX_LEN, MAX_LEN),
                   dtype=torch.float32).triu(diagonal=1)

    hidden_states = embeds(init_ids, init_pos).view(1, -1, hidden_size)
    print(hidden_states.size())
    past_layers = []
    for i in range(num_layers):
        hidden_states, kv_cache = blocks[i](hidden_states, attention_mask)
        kv_cache[:, MAX_LEN - token_len:, :] = kv_cache[:, :token_len, :]
        past_layers.append(kv_cache)

    token = get_token(lm_head, hidden_states[:, token_len - 1:token_len, :])

    result = []
    result.append(token.item())
    ng = 1
    token_len += 1

    while ng < 100:
        hidden_states = embeds(token, torch.tensor(
            token_len - 1)).view(1, -1, hidden_size)

        attention_mask = -1000 * torch.ones((1, MAX_LEN + 1))
        attention_mask[:, MAX_LEN - token_len + 1:] = 0

        attention_mask = attention_mask.to(torch.float32)

        for i in range(num_layers):
            hidden_states, current_cache = blocks_cache[i](
                hidden_states, past_layers[i], attention_mask)

            past_layers[i] = current_cache[:, 1:, :]

        token = get_token(lm_head, hidden_states)
        result.append(token.item())
        token_len += 1
        ng += 1

    return tokenizer.decode(result, skip_special_tokens=True)


def convert_block(layer_id):
    hidden_states = torch.rand((1, MAX_LEN, hidden_size))

    attention_mask = -1000 * \
        torch.ones((MAX_LEN, MAX_LEN),
                   dtype=torch.float32).triu(diagonal=1)
    model = Block(layer_id).eval()

    torch.onnx.export(
        model, (hidden_states, attention_mask),
        f'{onnx_model_path}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'attention_mask'],
        output_names=['hidden_states', 'past_layer'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    hidden_states = torch.rand((1, 1, hidden_size))
    past_layer = torch.rand((1, 512, 256))
    attention_mask = -1000 * torch.ones((1, MAX_LEN + 1))
    attention_mask[:, MAX_LEN - 99 + 1:] = 0

    model = BlockCache(layer_id).eval()

    torch.onnx.export(
        model, (hidden_states, past_layer, attention_mask),
        f'{onnx_model_path}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'past_cache', 'attention_mask'],
        output_names=['hidden_states', 'current_cache'],
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    ids = torch.tensor([[0, 1, 2, 3]])
    pids = torch.tensor([[0, 1, 2, 3]])

    dynamic_axes = {
        "input_ids": {1: "length"},
        "input_pos": {1: "length"}
    }
    torch.onnx.export(model, (ids, pids),
                      f'{onnx_model_path}/embeddings.onnx',
                      verbose=False,
                      input_names=['input_ids', 'input_pos'],
                      dynamic_axes=dynamic_axes,
                      output_names=['hidden_state'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(hidden_size)
    torch.onnx.export(model, (input),
                      f'{onnx_model_path}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)


if __name__ == "__main__":
    for i in range(num_layers):
        print(f'Convert {i}/{num_layers}\'s block...')
        convert_block(i)
        print(f'Convert {i}/{num_layers}\'s block cache...')
        convert_block_cache(i)
    print(f'Convert embedding')
    convert_embedding()
    print(f'Convert lm_head')
    convert_lm_head()
