#include "wizardcoder.h"
#include <bits/stdc++.h>
#include <cxxopts.hpp>
#include <utils.h>
#include "tokenizer.h"

int main(int argc, char** argv) {
    cxxopts::Options options(
            "Wizardcoder-TPU",
            "Wizardcoder-15B-V1.0 Implementation On Sophon BM1684X");

    options.add_options()(
            "m,model_path", "bmodel path", cxxopts::value<std::string>())(
            "n,new_length",
            "new max length",
            cxxopts::value<int>()->default_value("300"))(
            "c,custom_input",
            "custom_input",
            cxxopts::value<bool>()->default_value("false"))(
            "d,dev_id",
            "TPU device ids",
            cxxopts::value<std::vector<int>>()->default_value("0"))(
            "h,help", "Print usage");

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << '\n';
        return 0;
    }

    auto model_path = args["model_path"].as<std::string>();
    auto max_new_length = args["new_length"].as<int>();
    auto custom_input = args["custom_input"].as<bool>();
    auto dev_ids = args["dev_id"].as<std::vector<int>>();

    auto model = WizardCoderModel::from_pretrained(model_path, dev_ids);

    if (!model) std::cerr << "Error\n";

    std::cout << FRED("WizardCoder-15B-V1.0-BM1684X: ") << "Welcome!"
              << FBLU("\nUser: ") << std::flush;

    std::string input_str;

    if (!custom_input) {
        auto inputs = std::vector<const char*>{
                "Write a Python code to count 1 to 10.",
                "Write a Jave code to sum 1 to 10.",
                "Write a Rust code to find SCC.",
                "Write a Go code to find LCA",
        };

        int cnt = 0;
        while (cnt < inputs.size()) {
            input_str = inputs[cnt++];
            std::cout << input_str << '\n' << std::flush;
            std::cout << FRED("WizardCoder-15B-V1.0-BM1684X: ") << std::flush;
            auto prompt = model->build_prompt(input_str);
            auto input_ids = model->encode(prompt);
            model->stream_generate(input_ids, max_new_length);
            std::cout << FBLU("\n\nUser: ") << std::flush;
        }
    } else {
        while (1) {
            std::getline(std::cin, input_str);
            std::cout << FRED("WizardCoder-15B-V1.0-BM1684X: ") << std::flush;
            auto prompt = model->build_prompt(input_str);
            auto input_ids = model->encode(prompt);
            model->stream_generate(input_ids, max_new_length);
            std::cout << FBLU("\n\nUser: ") << std::flush;
        }
    }

    return 0;
}