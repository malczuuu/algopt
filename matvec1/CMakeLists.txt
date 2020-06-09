cmake_minimum_required(VERSION 3.10)
project(matvec1)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build/lib)

add_compile_options(-mavx -march=native)

set(MATVEC1_SOURCES src/algorithms1.cpp src/algorithms1.hpp src/matvec1.cpp)
add_executable(matvec1 ${MATVEC1_SOURCES})
