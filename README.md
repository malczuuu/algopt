# Algorithms optimization samples

[![Build Status](https://travis-ci.org/malczuuu/algopt.svg?branch=master)](https://travis-ci.org/malczuuu/algopt)

Optimization of algorithms for operations on vectors and matrices.

Projects from modeling of the technical problems classes at [Cracow University of Technology](https://pk.edu.pl) created
by Damian Malczewski in 2019/20.

Project structure was converted from Visual Studio into CMake project to be able to build and run on other platforms as 
well.

* [`matvec1`](/src/matvec1) - matrix by vector multiplication (1)
* [`matvec2`](/src/matvec2) - matrix by vector multiplication (2)
* [`dotprod`](/src/dotprod) - dot product of two vectors

## Build and run

* `cmake .` to generate `Makefile` and initialize project files
* `make` to build sources, output files will be located within `./build/bin/` directory

## `matvec1`

Benchmark tool for matrix by vector multiplication. Matrix is stored within single-dimensional array column-by-column.

* naive algorithm
* fixed memory jumps
* 8-times loop unwinding
* SSE2 vectorization (Streaming SIMD Extensions 2)
* SSE2 with loop unwinding
* AVX vectorization (Advanced Vector Extensions)
* AVX with FMA instruction set

## `matvec2`

Benchmark tool for matrix by vector multiplication. Matrix is stored within single-dimensional array row-by-row.

* naive algorithm
* SSE2 with loop unwinding
* AVX with loop unwinding
* AVX+FMA with loop unwinding

## `dotprod`

Benchmark tool for dot product calculation in multi-threaded mode.

* naive algorithm
* SSE2 with loop unwinding
* AVX with loop unwinding
* AVX+FMA with loop unwinding

## Code style

* Code should be formatted using `clang-format -i src/**/*.cpp src/**/*.hpp` command.
