cmake_minimum_required(VERSION 3.10)
project(algopt)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build/lib)

add_compile_options(-mavx -march=native)


set(DOT_PROD_SOURCES
        src/dotprod/dot_data_t.hpp
        src/dotprod/dot_prod_main.cpp
        src/dotprod/thread_funcs.hpp
        src/dotprod/thread_funcs.cpp
        src/dotprod/utility.hpp
        src/dotprod/utility.cpp)
add_executable(dotprod ${DOT_PROD_SOURCES})
target_link_libraries(dotprod ${CMAKE_THREAD_LIBS_INIT})


set(MATVEC1_SOURCES
        src/matvec1/algorithms.cpp
        src/matvec1/algorithms.hpp
        src/matvec1/matvec1_main.cpp
        src/matvec1/utility.cpp
        src/matvec1/utility.hpp)
add_executable(matvec1 ${MATVEC1_SOURCES})


set(MATVEC2_SOURCES
        src/matvec2/algorithms.cpp
        src/matvec2/algorithms.hpp
        src/matvec2/matvec2_main.cpp
        src/matvec2/utility.cpp
        src/matvec2/utility.hpp)
add_executable(matvec2 ${MATVEC2_SOURCES})
