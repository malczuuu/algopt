cmake_minimum_required(VERSION 3.10)
project(algopt)

set(CMAKE_CXX_STANDARD 14)

find_package (Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

add_compile_options(-mavx -march=native)


set(DOT_PROD_SOURCES
        dot-prod/dot_data_t.hpp
        dot-prod/dot_prod_main.cpp
        dot-prod/thread_funcs.hpp
        dot-prod/thread_funcs.cpp
        dot-prod/utility.hpp
        dot-prod/utility.cpp)

add_executable(dotprod ${DOT_PROD_SOURCES})

target_link_libraries (dotprod ${CMAKE_THREAD_LIBS_INIT})


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
