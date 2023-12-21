![image](./assets/sophgo_chip.png)

# OPT-TPU

本项目实现BM1684X部署语言大模型[OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


## Build


### Requirements

- 支持C++20标准的gcc或clang编译器（demo本身只需要C++17，依赖ctre需要C++20）
- libsophon库（默认在```/opt/sophon/libsophon-current```，如果不在默认路径或者需要特定版本的libsophon，需要在下面编译时指定```LIBSOPHON_DIR```）
- 转换好的OPT-6.7 bmodel文件，和huggingface仓库中clone的```vocab.json```和```merges.txt```放在一起。最好直接```GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/facebook/opt-6.7b && cp *.bmodel opt-6.7b```将它们放在一起
### Build

```shell
cd build
cmake .. -GNinja -DLIBSOPHON_DIR=...
ninja
```
### Inference

完成上文的编译过程后，生成```build/demo/demo```可执行文件，它可以完成加载bmodel并在```BM1684X```设备上进行推理。执行```build/demo/demo -h```可以查看参数含义。
加上```-c```表明使用用户输入，否则会自动使用默认的输入展示

## 模型转换



### Prepreparation

修改```transformers.models.opt.modeling_opt```中

```python
class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)

```
的```OPTLearnedPositionalEmbedding```如下：
```python
class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    # def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
    #     """`input_ids_shape` is expected to be [bsz x seqlen]."""
    #     attention_mask = attention_mask.long()

    #     # create positions depending on attention_mask
    #     positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

    #     # cut positions if `past_key_values_length` is > 0
    #     positions = positions[:, past_key_values_length:]

    #     return super().forward(positions + self.offset)
    def forward(self, position_ids):
        return super().forward(position_ids + self.offset)

```
在这里为了方便我们手动传入```position_id```，而不是由```attention_mask```计算
### 模型转换
```shell
python export_to_onnx.py
./compile.sh --mode int8
```
稍等片刻即可导出int8量化后的bmodel文件，若需要int4量化，使用```./compile.sh --mode int4```即可


由于脚本中有很多相对路径，一个供参考的目录layout为：


![Alt text](/assets/image.png)