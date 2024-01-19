project(pybmwizardcoder)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(SWIG REQUIRED COMPONENTS python)
include(UseSWIG)
set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)

macro(configure_pybmwizardcoder source)
    set_source_files_properties(${source} PROPERTIES
        CPLUSPLUS ON
        USE_TARGET_INCLUDE_DIRECTORIES TRUE
    )
endmacro()

swig_add_library(pybmwizardcoder
    TYPE SHARED
    LANGUAGE python
    SOURCES pybmwizardcoder.i
)
set_property(TARGET pybmwizardcoder PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET pybmwizardcoder PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)

set_source_files_properties(pybmwizardcoder.i PROPERTIES
    CPLUSPLUS ON
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/../c/
)

target_include_directories(pybmwizardcoder PRIVATE  ${PROJECT_SOURCE_DIR}/../c/)
target_link_directories(pybmwizardcoder PRIVATE ${LIBSOPHON_DIR}/lib)


configure_pybmwizardcoder(pybmwizardcoder.i)


target_link_libraries(pybmwizardcoder PRIVATE
    bmwizardcoder
    Python::Module
)