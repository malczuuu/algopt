# Algorithms optimization samples

[![Build Status](https://travis-ci.org/malczuuu/algopt.svg?branch=master)](https://travis-ci.org/malczuuu/algopt)

Optimization of algorithms for operations on vectors and matrices.

* [`matvec1`](/src/matvec1) - matrix by vector multiplication (1)
* [`matvec2`](/src/matvec2) - matrix by vector multiplication (2)
* [`dotprod`](/src/dotprod) - dot product of two vectors

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
