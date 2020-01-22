cmake_minimum_required(VERSION 3.10)
project(algopt)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

set(CMAKE_CXX_FLAGS -mavx)
set(CMAKE_CXX_FLAGS -march=native)


set(MATVEC1_SOURCES
        matvec1/algorithms.cpp
        matvec1/algorithms.hpp
        matvec1/matvec1_main.cpp
        matvec1/utility.cpp
        matvec1/utility.hpp)

add_executable(matvec1 ${MATVEC1_SOURCES})


set(MATVEC2_SOURCES
        matvec2/algorithms.cpp
        matvec2/algorithms.hpp
        matvec2/matvec2_main.cpp
        matvec2/utility.cpp
        matvec2/utility.hpp)

add_executable(matvec2 ${MATVEC2_SOURCES})
