#include <simdjson.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <ostream>
#include "include/config.h"

namespace fs = std::filesystem;
using namespace simdjson;

std::optional<AutoConfig> AutoConfig::from_pretrained(
        std::string_view model_path) {
    fs::path path(model_path);
    auto     config_file = path.parent_path() / "config.json";

    std::ifstream ifs(config_file);
    if (!ifs.good()) {
        std::cerr << "open config.json error\n";
        return std::nullopt;
    }

    auto cfg = AutoConfig();

    ondemand::parser parser;
    padded_string    config_data = padded_string::load(config_file.c_str());
    auto             config = parser.iterate(config_data);

    cfg.num_layers = config["n_layer"].get_int64();
    cfg.num_heads = config["n_head"].get_int64();
    cfg.hidden_size = config["n_embd"].get_int64();
    return cfg;
}
