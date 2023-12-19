![image](./assets/sophgo_chip.png)

# OPT-TPU

本项目实现BM1684X部署语言大模型[OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。
