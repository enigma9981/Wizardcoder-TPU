#include "wizardcoder_c.h"
#include <string>
#include <vector>
#include "wizardcoder.h"
// #include "wizardcoder.h"

struct BMWizardCoder {
    WizardCoderModel* inner;
};

std::string res;
std::string tmp;

extern "C" BMWizardCoder* bmwizardcoder_create() {
    BMWizardCoder* model = new BMWizardCoder;
    model->inner = new WizardCoderModel;
    return model;
}

extern "C" void bmwizardcoder_init(
        BMWizardCoder* instance,
        int*           devids,
        int            num_device,
        const char*    model_dir) {
    std::vector<int> ids(devids, devids + num_device);
    instance->inner->init(model_dir, ids);
}

extern "C" void bmwizardcoder_stream_complete(
        BMWizardCoder* instance,
        const char*    input,
        int            max_new_length) {
    tmp = std::string(input);
    auto prompt = instance->inner->build_prompt(tmp);
    auto ids = instance->inner->encode(prompt);
    instance->inner->stream_generate(ids, max_new_length);
}

extern "C" const char* bmwizardcoder_complete(
        BMWizardCoder* instance,
        const char*    input,
        int            max_new_length) {
    tmp = std::string(input);
    auto prompt = instance->inner->build_prompt(tmp);
    auto ids = instance->inner->encode(prompt);
    res = instance->inner->generate(ids, max_new_length);
    return res.c_str();
}