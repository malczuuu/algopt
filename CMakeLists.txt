cmake_minimum_required(VERSION 3.10)
project(algopt)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build/lib)

add_compile_options(-mavx -march=native)


set(DOTPROD_SOURCES
        src/dotprod/dot_data_t.hpp
        src/dotprod/dot_prod_main.cpp
        src/dotprod/dot_prod_funcs.hpp
        src/dotprod/dot_prod_funcs.cpp
        )
add_executable(dotprod ${DOTPROD_SOURCES})
target_link_libraries(dotprod ${CMAKE_THREAD_LIBS_INIT})


set(MATVEC1_SOURCES
        src/matvec1/matvec1_algorithms.cpp
        src/matvec1/matvec1_algorithms.hpp
        src/matvec1/matvec1_main.cpp
        )
add_executable(matvec1 ${MATVEC1_SOURCES})


set(MATVEC2_SOURCES
        src/matvec2/matvec2_algorithms.cpp
        src/matvec2/matvec2_algorithms.hpp
        src/matvec2/matvec2_main.cpp
        )
add_executable(matvec2 ${MATVEC2_SOURCES})
