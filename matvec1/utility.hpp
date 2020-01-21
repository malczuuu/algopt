#ifndef __MATVEC1_UTILITY__
#define __MATVEC1_UTILITY__

#include <vector>

std::vector<int> parseargs(int argc, char *argv[]);

long instant();

void prepare(double *a_matr, double *x_vect, int size);

void prepare_block(double *a_matr, double *x_vect, int size, int block_size);

#endif
