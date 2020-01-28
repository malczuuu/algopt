#ifndef __MATVEC2_ALGORITHMS__
#define __MATVEC2_ALGORITHMS__

void matvec2_naive(double* y_vect, const double* a_matr, const double* x_vect, int size);

void matvec2_sse2(double* y_vect, const double* a_matr, const double* x_vect, int size);

void matvec2_avx(double* y_vect, const double* a_matr, const double* x_vect, int size);

void matvec2_fma_avx(double* y_vect, const double* a_matr, const double* x_vect, int size);

#endif
