![image](./assets/sophgo_chip.png)

# Wizardcoder-TPU

本项目实现BM1684X部署语言大模型[Wizardcoder-15B](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


## Build


### Requirements

- 支持C++20标准的gcc或clang编译器
- 如果不使用```demo/libsophon_pcie```的libsophon或者需要特定版本的libsophon，需要在下面编译时指定```LIBSOPHON_DIR```
- 转换好的Wizardcoder-15B.bmodel文件，直接```GIT_LFS_SKIP_SMUDGE=1 git clone git clone https://huggingface.co/WizardLM/WizardCoder-15B-V1.0 && cp *.bmodel WizardCoder-15B-V1.0```将它们放在一起。demo会使用原仓库下的配置和tokenizer配置
- 如果需要在```python```下直接推理或者evalatuion，需要```python```，版本不低于3.8其他和依赖 ```libpython3-dev, python3-numpy, swig```
### Build

```shell
cd build
cmake .. -GNinja -DLIBSOPHON_DIR=...
ninja
```
### Inference

#### C++
完成上文的编译过程后，生成```build/demo/bin/wizardcodercc```可执行文件，它可以完成加载bmodel并在```BM1684X```设备上进行推理。执行```build/demo/demo -h```可以查看参数含义。
加上```-c```表明使用用户输入，否则会自动使用默认的输入展示



## 模型转换


### 环境
- python >= 3.11
- torch >= 2.1



### 修改模型文件

- 使用```pip show transformers```找到```transformers```的位置
- 使用提供的```modeling_gpt_bigcode.py```替换```python3.11/site-packages/transformers/models/gpt_bigcode/```下的同名文件


### 模型转换

```shell
python export_to_onnx.py
./compile.sh --mode int4
```
稍等片刻即可导出int4量化后的bmodel文件
