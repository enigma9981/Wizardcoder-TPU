if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH local)
endif()


if(NOT DEFINED LIBSOPHON_DIR)
    set(LIBSOPHON_DIR ${CMAKE_SOURCE_DIR}/demo/libsophon_pcie CACHE STRING "The directory of libsophon")
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


FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
)



FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)


FetchContent_MakeAvailable(ctre cxxopts simdjson cnpy json)


get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()



# SET(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)


include_directories(${LIBSOPHON_DIR}/include)
file(GLOB OPT_SRC gpt2tokenizer.cc wizardcoder.cc opt.cc)
add_library(tpuopt ${OPT_SRC})
target_include_directories(tpuopt PRIVATE ${CNPY_SOURCE_DIR})
target_link_directories(tpuopt PRIVATE ${LIBSOPHON_DIR}/lib)

# target_include_directories(tpuopt PRIVATE ${ONNXRUNTIME_DIR}/include)
# target_link_directories(tpuopt PRIVATE ${ONNXRUNTIME_DIR}/lib)

target_link_libraries(tpuopt PRIVATE simdjson::simdjson ctre bmlib bmrt cnpy)



add_subdirectory(bin)
