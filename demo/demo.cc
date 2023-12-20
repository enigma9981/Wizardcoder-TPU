#include <bits/stdc++.h>
#include <cxxopts.hpp>
#include "include/opt.h"
#include "include/tokenizer.h"

int main(int argc, char** argv) {
    cxxopts::Options options(
            "OPT-TPU", "OPT-6.7B Implementation On Sophon BM1684X");

    options.add_options()(
            "m,model_path", "bmodel path", cxxopts::value<std::string>())(
            "n,new_length",
            "new max length",
            cxxopts::value<int>()->default_value("20"))(
            "c,custom_input",
            "custom_input",
            cxxopts::value<bool>()->default_value("false"))(
            "d,dev_id",
            "TPU device id",
            cxxopts::value<int>()->default_value("0"))("h,help", "Print usage");
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << '\n';
        return 0;
    }

    auto model_path = result["model_path"].as<std::string>();
    auto max_new_length = result["new_length"].as<int>();
    auto custom_input = result["custom_input"].as<bool>();
    auto dev_id = result["dev_id"].as<int>();

    auto model = OPTModel::from_pretrained(model_path, dev_id);
    if (!model) {
        std::cerr << "Load model error\n";
        return -1;
    }

    if (custom_input) {
        std::cout << "OPT on BM1684X\n";
        while (1) {
            std::cout << "> ";
            std::string input_string;
            std::getline(std::cin, input_string);
            auto ids = model->encode(input_string);
            std::cout << "< " << std::flush;
            model->stream_generate(ids, max_new_length);
            std::cout << "\n\n" << std::flush;
        }

    } else {
        std::cout << "No user input, using builtin prompts:\n";
        auto prompts = std::vector<std::string>{
                "Question: if x is 2 and y is 5, what is x + y?\nAnswer: 7\n\nQuestion: if x is 12 and y is 9, what is x + y?\nAnswer: ",
                "What is the color of a carrot?\nA:",
                "What are we having for dinner?",
                "My name is Linda ",
                "Hey, are you conscious? Can you talk to me?",
                "A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived there?"};
        int cnt = 0;
        for (auto&& e : prompts) {
            std::cout << "Example " << cnt++ << ":\n" << e << std::flush;
            auto ids = model->encode(e);
            model->stream_generate(ids, max_new_length);
            std::cout << "\n\n";
        }
    }

    return 0;
}