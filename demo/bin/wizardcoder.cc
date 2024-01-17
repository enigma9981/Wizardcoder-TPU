#include "wizardcoder.h"
#include <bits/stdc++.h>
#include <cxxopts.hpp>
#include <utils.h>
#include <bitset>
#include <chrono>
#include <ostream>
#include <string>
#include <vector>
#include "opt.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    std::vector<int> ids{0};
    auto             model = WizardCoderModel::from_pretrained(argv[1], ids);

    if (!model) std::cerr << "Error\n";

    std::cout << FRED("WizardCoder-15B-V1.0-BM1684X: ") << "Welcome!"
              << FBLU("\nUser: ") << std::flush;

    std::string input_str;

    while (1) {
        std::getline(std::cin, input_str);
        // input_str = "Write a Python code to count 1 to 10.";
        std::cout << FRED("WizardCoder-15B-V1.0-BM1684X: ") << std::flush;
        auto prompt = model->build_prompt(input_str);
        auto input_ids = model->encode(prompt);
        model->stream_generate(input_ids, 500);
        std::cout << FBLU("\n\nUser: ") << std::flush;
    }

    return 0;
}