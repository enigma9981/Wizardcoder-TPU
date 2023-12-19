if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH local)
endif()


if(NOT DEFINED LIBSOPHON_DIR)
    set(LIBSOPHON_DIR "/opt/sophon/libsophon-current" CACHE STRING "The directory of libsophon")
endif()
message(STATUS "LIBSOPHON_DIR is set to: ${LIBSOPHON_DIR}")


include(FetchContent)


FetchContent_Declare(
  ctre
  GIT_REPOSITORY https://github.com/hanickadot/compile-time-regular-expressions.git
  GIT_TAG        v3.8.1
)


FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.1.1
)

FetchContent_Declare(
  simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG        v3.6.3
)

FetchContent_MakeAvailable(ctre cxxopts simdjson)


file(GLOB OPT_SRC gpt2tokenizer.cc opt.cc)
add_library(tpuopt ${OPT_SRC})
target_link_directories(tpuopt PRIVATE ${LIBSOPHON_DIR}/lib)
target_link_libraries(tpuopt PRIVATE simdjson::simdjson ctre bmlib bmrt)


file(GLOB DEMO_SRC demo.cc)
add_executable(demo ${DEMO_SRC})
include_directories(${LIBSOPHON_DIR}/include)
target_link_directories(demo PRIVATE ${LIBSOPHON_DIR}/lib)
target_link_libraries(demo PRIVATE tpuopt cxxopts)




