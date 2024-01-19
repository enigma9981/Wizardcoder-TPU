

add_library(bmwizardcoder  wizardcoder_c.cc)

target_link_libraries(bmwizardcoder PUBLIC tpuopt)

target_include_directories(bmwizardcoder PRIVATE ${PROJECT_SOURCE_DIR}/demo/include)




add_executable(bmwizardcoder_test bmwizardcoder_test.c)
target_link_directories(bmwizardcoder_test PRIVATE ${LIBSOPHON_DIR}/lib)
target_link_libraries(bmwizardcoder_test PRIVATE bmwizardcoder)