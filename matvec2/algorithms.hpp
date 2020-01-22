#ifndef __MATVEC2_ALGORITHMS__
#define __MATVEC2_ALGORITHMS__

void matvec_naive(double* y_vect, const double* a_matr, const double* x_vect, int size);

void matvec_sse2(double* y_vect, const double* a_matr, const double* x_vect, int size);

void matvec_avx(double* y_vect, const double* a_matr, const double* x_vect, int size);

void matvec_fma_avx(double* y_vect, const double* a_matr, const double* x_vect, int size);

#endif
