cmake_minimum_required(VERSION 3.14)
project(sg_accel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)
if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find pybind11. Install with: pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(sg_accel sg_accel.cpp)
target_compile_options(sg_accel PRIVATE -O3 -march=native -ffast-math -DNDEBUG)
install(TARGETS sg_accel DESTINATION .)
