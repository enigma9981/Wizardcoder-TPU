![image](./assets/sophgo_chip.png)

# OPT-TPU

本项目实现BM1684X部署语言大模型[OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

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