#ifndef CONFIG_H
#define CONFIG_H

// struct

#include <cstddef>
#include <string_view>
struct AutoConfig {
    int               num_layers;
    int               num_heads;
    int               hidden_size;
    int               bmodel_max_len;
    static AutoConfig from_pretrained(std::string_view model_path);
};

#endif