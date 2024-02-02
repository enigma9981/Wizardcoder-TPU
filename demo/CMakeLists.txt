cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
project(wizardocder-tpu)


if(NOT DEFINED LIBSOPHON_DIR)
    set(LIBSOPHON_DIR ${CMAKE_SOURCE_DIR}/libsophon_pcie CACHE STRING "The directory of libsophon")
endif()
message(STATUS "LIBSOPHON_DIR is set to: ${LIBSOPHON_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")

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


FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
)


FetchContent_MakeAvailable(simdjson cnpy ctre)



add_executable(wizardcoder demo.cpp gpt2tokenizer.cc)
target_include_directories(wizardcoder PRIVATE ${LIBSOPHON_DIR}/include)
target_link_directories(wizardcoder PRIVATE ${LIBSOPHON_DIR}/lib)
target_link_libraries(wizardcoder PRIVATE bmrt bmlib simdjson::simdjson ctre cnpy)
