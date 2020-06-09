cmake_minimum_required(VERSION 3.10)
project(matvec2)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build/lib)

add_compile_options(-mavx -march=native)

set(MATVEC2_SOURCES src/algorithms2.cpp src/algorithms2.hpp src/matvec2.cpp)
add_executable(matvec2 ${MATVEC2_SOURCES})
