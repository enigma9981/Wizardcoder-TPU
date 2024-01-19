#ifndef WIZARDCODER_C_H
#define WIZARDCODER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#define EXPORT_BM_PREFIX(func) bmwizardcoder_##func
typedef struct BMWizardCoder BMWizardCoder;

BMWizardCoder* EXPORT_BM_PREFIX(create)();

// void EXPORT_BM_PREFIX(destory)(BMWizardCoder* instance);

void EXPORT_BM_PREFIX(init)(
        BMWizardCoder* instance,
        int*           devids,
        int            num_device,
        const char*    model_dir);

void EXPORT_BM_PREFIX(stream_complete)(
        BMWizardCoder* instance,
        const char*    input,
        int            max_new_length);

const char* EXPORT_BM_PREFIX(complete)(
        BMWizardCoder* instance,
        const char*    input,
        int            max_new_length);

#ifdef __cplusplus
}
#endif

#endif