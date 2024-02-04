![image](./assets/sophgo_chip.png)

# Wizardcoder-TPU

本项目实现BM1684X部署语言大模型[Wizardcoder-15B](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


## Build


### Requirements

- 支持C++20标准的gcc或clang编译器
- 如果不使用```demo/libsophon_pcie```的libsophon或者需要特定版本的libsophon，需要在下面编译时指定```LIBSOPHON_DIR```
- 转换好的Wizardcoder-15B.bmodel文件，需要和本仓库中```vocab```目录下的两个文件放在一起。模型转换过程可以参考下文

### Build

```shell
mkdir demo/build
cd demo/build
cmake .. 
make
```
### Inference

#### C++
完成上文的编译过程后，生成```demo/build/wizardcoder```可执行文件，它可以完成加载bmodel并在```BM1684X```设备上进行推理。示例：

```shell
demo/build/wizardcoder -m /path/to/bmodel -d 0
```

- -m 指定bmodel的位置，在bmodel的同级目录下，需要有```vovab```目录的两个```vocab.json```和```merges.txt```，作为tokenzier需要的文件
- -d 指定推理使用的芯片，默认id是0，如果需要在多芯片上推理，请使用逗号分割，如：-d 0,1,2,3

## 模型转换


### 环境
- python >= 3.11
- torch >= 2.1
python3.10和较低版本的torch可能会导致一些预想外的错误，建议使用较新的3.11和2.1版本的torch

### 修改模型文件
- 使用```pip show transformers```找到```transformers```的位置
- 使用提供的```modeling_gpt_bigcode.py```替换```python3.11/site-packages/transformers/models/gpt_bigcode/```下的同名文件

### 模型转换
#### 导出ONNX格式的模型
```shell
python export_to_onnx.py --model_path your_model_path
```
- 本步骤可以在docker外进行
- your_model_path 指的是原模型下载后的地址, 如:"../../WizardCoder-15B-V1.0/"
- 如果你想要debug，而不是一下子生成完成全部的onnx模型，可以将36行的num_layers改成1, 结合144行的函数对比单个block情况下是否可以和pytroch版本对齐
- 脚本运行完毕后会在```./tmp/```下生成大量ONNX模型，用于后续在TPU上进行转换
#### 模型编译
```shell
./compile.sh --mode int4
```
稍等片刻即可导出int4量化后的bmodel文件
- 模型编译必须要在docker内完成，无法在docker外操作，准备相应docker和tpu-mlir环境，可以参考Llama-TPU的做法：
[Llama2-TPU](https://github.com/sophgo/Llama2-TPU/blob/main/README.md)
- 受TPU内存的限制```./compile.sh```目前仅支持在单芯上进行INT4量化，执行```./compile.sh --mode int4```后会在目录下生成```wizardcoder-15B_int4.bmodel```文件
- 后续会支持多芯上的推理，请使用```./compile.sh --mode [F16/int8/int4] --num_device [2/4/8]```进行转换，编译完成后最终会在compile路径下生成名为wizardcoder-15B_{X}_{Y}dev.bmodel的模型文件，其中X代表量化方式，其值有F16/int8/int4等，Y代表使用的芯片个数，其值可能有1/2/4/8等。
- 生成bmodel耗时大概3小时以上，建议64G内存以及300G以上磁盘空间，不然很可能OOM或者no space left

