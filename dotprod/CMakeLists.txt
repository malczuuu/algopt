cmake_minimum_required(VERSION 3.10)
project(dotprod)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build/lib)

add_compile_options(-mavx -march=native)

set(DOTPROD_SOURCES src/dot_data_t.hpp src/dotprod.cpp src/dot_prod_funcs.hpp src/dot_prod_funcs.cpp)
add_executable(dotprod ${DOTPROD_SOURCES})
target_link_libraries(dotprod ${CMAKE_THREAD_LIBS_INIT})
