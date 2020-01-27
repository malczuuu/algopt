#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>

void matvec_naive(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    for (int i = 0; i < size; ++i) {
        y_vect[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            y_vect[i] += a_matr[i * size + j] * x_vect[j];
        }
    }
}

void matvec_sse2(double* y_vect, const double* a_matr, const double* x_vect, int size) {}

void matvec_avx(double* y_vect, const double* a_matr, const double* x_vect, int size) {}

void matvec_fma_avx(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    int outer_unwinding = 4;
    int inner_unwinding = 16;

    __m256d ry0, ry1, ry2, ry3;
    __m256d rx0, rx1, rx2, rx3;
    __m256d ra0, ra1, ra2, ra3;

    const double* ptr_a = a_matr;
    const double* ptr_x = nullptr;

    double* buf0 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));
    double* buf1 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));
    double* buf2 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));
    double* buf3 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));

    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    for (int i = 0; i < size; i += outer_unwinding) {
        ry0 = ry1 = ry2 = ry3 = _mm256_setzero_pd();
        ptr_x = x_vect;

        for (int j = 0; j < size; j += inner_unwinding) {
            _mm_prefetch(static_cast<const void*>(ptr_x + inner_unwinding), _MM_HINT_T0);
            _mm_prefetch(static_cast<const void*>(ptr_x + inner_unwinding + 8), _MM_HINT_T0);
            _mm_prefetch(static_cast<const void*>(ptr_a + inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + inner_unwinding + 8), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + size + inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + size + inner_unwinding + 8), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + 2 * size + inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + 2 * size + inner_unwinding + 8), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + 3 * size + inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + 3 * size + inner_unwinding + 8), _MM_HINT_NTA);

            rx0 = _mm256_load_pd(ptr_x);
            rx1 = _mm256_load_pd(ptr_x + 4);
            rx2 = _mm256_load_pd(ptr_x + 8);
            rx3 = _mm256_load_pd(ptr_x + 12);

            ra0 = _mm256_load_pd(ptr_a);
            ra1 = _mm256_load_pd(ptr_a + size);
            ra2 = _mm256_load_pd(ptr_a + 2 * size);
            ra3 = _mm256_load_pd(ptr_a + 3 * size);

            ry0 = _mm256_fmadd_pd(ra0, rx0, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx0, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx0, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx0, ry3);

            ra0 = _mm256_load_pd(ptr_a + 4);
            ra1 = _mm256_load_pd(ptr_a + 4 + size);
            ra2 = _mm256_load_pd(ptr_a + 4 + 2 * size);
            ra3 = _mm256_load_pd(ptr_a + 4 + 3 * size);

            ry0 = _mm256_fmadd_pd(ra0, rx1, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx1, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx1, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx1, ry3);

            ra0 = _mm256_load_pd(ptr_a + 8);
            ra1 = _mm256_load_pd(ptr_a + 8 + size);
            ra2 = _mm256_load_pd(ptr_a + 8 + 2 * size);
            ra3 = _mm256_load_pd(ptr_a + 8 + 3 * size);

            ry0 = _mm256_fmadd_pd(ra0, rx2, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx2, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx2, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx2, ry3);

            ra0 = _mm256_load_pd(ptr_a + 12);
            ra1 = _mm256_load_pd(ptr_a + 12 + size);
            ra2 = _mm256_load_pd(ptr_a + 12 + 2 * size);
            ra3 = _mm256_load_pd(ptr_a + 12 + 3 * size);

            ry0 = _mm256_fmadd_pd(ra0, rx3, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx3, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx3, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx3, ry3);

            ptr_a += inner_unwinding;
            ptr_x += inner_unwinding;
        }

        ptr_a += (outer_unwinding - 1) * size;

        _mm256_store_pd(buf0, ry0);
        _mm256_store_pd(buf1, ry1);
        _mm256_store_pd(buf2, ry2);
        _mm256_store_pd(buf3, ry3);

        y_vect[i] = buf0[0] + buf0[1] + buf0[2] + buf0[3];
        y_vect[i + 1] = buf1[0] + buf1[1] + buf1[2] + buf1[3];
        y_vect[i + 2] = buf2[0] + buf2[1] + buf2[2] + buf2[3];
        y_vect[i + 3] = buf3[0] + buf3[1] + buf3[2] + buf3[3];
    }

    free(buf0);
    free(buf1);
    free(buf2);
    free(buf3);
}
