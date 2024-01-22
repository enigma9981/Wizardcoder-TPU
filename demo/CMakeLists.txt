if(NOT DEFINED LIBSOPHON_DIR)
    set(LIBSOPHON_DIR ${CMAKE_SOURCE_DIR}/demo/libsophon_pcie CACHE STRING "The directory of libsophon")
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


FetchContent_MakeAvailable(ctre cxxopts simdjson cnpy)



include_directories(${LIBSOPHON_DIR}/include)
file(GLOB OPT_SRC *.cc)
message(STATUS ${OPT_SRC})
add_library(tpuopt ${OPT_SRC})
target_include_directories(tpuopt PRIVATE ${CNPY_SOURCE_DIR})
target_link_directories(tpuopt PRIVATE ${LIBSOPHON_DIR}/lib)



target_link_libraries(tpuopt PRIVATE bmrt bmlib simdjson::simdjson ctre cnpy)

add_subdirectory(bin)


if(BUILD_PYTHON)
  add_subdirectory(c)
  add_subdirectory(python)
endif()