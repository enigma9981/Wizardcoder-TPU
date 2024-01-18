#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>
#include <optional>
#include <string_view>
struct AutoConfig {
    AutoConfig() = default;

    int num_layers;
    int num_heads;
    int hidden_size;
    int bmodel_max_len;

    static std::optional<AutoConfig> from_pretrained(
            std::string_view model_path);
};

#endif