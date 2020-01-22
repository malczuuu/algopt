#ifndef __MATVEC2_UTILITY__
#define __MATVEC2_UTILITY__

#include <vector>

std::vector<int> parseargs(int argc, char* argv[]);

long instant();

void prepare(double* a_matr, double* x_vect, int size);

#endif
