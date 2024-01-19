#ifndef WIZARDCODER_C_H
#define WIZARDCODER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#define EXPORT_BM(func) bmwizardcoder_##func
typedef struct BMWizardCoder BMWizardCoder;

BMWizardCoder* EXPORT_BM(create)();

// void EXPORT_BM(destory)(BMWizardCoder* instance);

void EXPORT_BM(init)(
        BMWizardCoder* instance,
        int*           devids,
        int            num_device,
        const char*    model_dir);

void EXPORT_BM(stream_complete)(
        BMWizardCoder* instance,
        const char*    input,
        int            max_new_length);

#ifdef __cplusplus
}
#endif

#endif