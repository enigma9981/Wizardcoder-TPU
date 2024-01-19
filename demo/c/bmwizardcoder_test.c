#include <stdio.h>
#include <string.h>
#include "wizardcoder_c.h"

int main(int argc, char* argv[]) {
    int devid[] = {0};

    BMWizardCoder* model = bmwizardcoder_create();
    bmwizardcoder_init(model, devid, 1, argv[1]);
    const char* test_input = "Write a Rust code to find SCC.";
    bmwizardcoder_stream_complete(model, test_input, 30);

    return 0;
}