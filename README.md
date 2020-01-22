# Algorithms optimization samples

[![Build Status](https://travis-ci.org/malczuuu/algopt.svg?branch=master)](https://travis-ci.org/malczuuu/algopt)

Optimization of algorithms for operations on vectors and matrices.

* [`matvec1`](/matvec1) - matrix by vector multiplication (1)
* [`matvec2`](/matvec2) - matrix by vector multiplication (2)

## `matvec1`

Benchmark tool for matrix by vector multiplication. Matrix is stored within single-dimensional array column-by-column.

* naive
* fixed memory jumps
* 8-times loop unwinding
* SSE2 vectorization (Streaming SIMD Extensions 2)
* AVX vectorization (Advanced Vector Extensions)

## Code style

* Code should be formatted using `clang-format -i **/*.cpp **/*.hpp` command.
