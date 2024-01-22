![image](./assets/sophgo_chip.png)

# Wizardcoder-TPU

本项目实现BM1684X部署语言大模型[Wizardcoder-15B](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


## Build


### Requirements

- 支持C++20标准的gcc或clang编译器
- 如果不使用```demo/libsophon_pcie```的libsophon或者需要特定版本的libsophon，需要在下面编译时指定```LIBSOPHON_DIR```）
- 转换好的Wizardcoder-15B.bmodel文件，直接```GIT_LFS_SKIP_SMUDGE=1 git clone git clone https://huggingface.co/WizardLM/WizardCoder-15B-V1.0 && cp *.bmodel WizardCoder-15B-V1.0```将它们放在一起。demo会使用原仓库下的配置和tokenizer配置
- 如果需要在```python```下直接推理或者evalatuion，需要```python```，版本不低于3.8其他和依赖 ```libpython3-dev, python3-numpy, swig```等，一个参考的编译环境为:
```d```
### Build

```shell
cd build
cmake .. -GNinja -DLIBSOPHON_DIR=...
ninja
```
### Build Python Binding
```shell
cd build
cmake .. -GNinja -DBUILD_PYTHON=ON 
ninja
```

### Inference

#### C++
完成上文的编译过程后，生成```build/demo/bin/wizardcodercc```可执行文件，它可以完成加载bmodel并在```BM1684X```设备上进行推理。
此时直接使用```demo/bin/wizardcodercc /path/to/.bmodel```即可在单颗芯片上完成推理

#### Python
完成上文的编译过程后，会在```/build/demo/python```下生成相关的python库，此时可以在Python中进行wizardcoder-15B的推理，参考示例:
```python
import pybmwizardcoder
import numpy as np

model_path = path/to/wizardcoder.bmodel

model = pybmwizardcoder.bmwizardcoder_create()
devs = np.array([0]).astype(np.int32)
pybmwizardcoder.bmwizardcoder_init(model, model_dir=model_path, devids=devs)


prompt = "Write a Rust code to find SCC."

x = pybmwizardcoder.bmwizardcoder_stream_complete(model, prompt, 20)

print(f'Result: {x}')
```

## 模型转换



### Prepreparation
### 转换和量化

