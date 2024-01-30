

include_directories(${LIBSOPHON_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/demo/include)

file(GLOB WIZARDCODERCC_SRC wizardcoder.cc)
add_executable(wizardcodercc ${WIZARDCODERCC_SRC})

target_link_directories(wizardcodercc PRIVATE ${LIBSOPHON_DIR}/lib)
target_link_libraries(wizardcodercc PRIVATE tpuopt cxxopts)

