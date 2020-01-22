#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>

void matvec_naive(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    for (int i = 0; i < size; ++i) {
        y_vect[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            y_vect[i] += a_matr[i + size * j] * x_vect[j];
        }
    }
}

void matvec_fixed_memjumps(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            y_vect[i] += a_matr[i + size * j] * x_vect[j];
        }
    }
}

void matvec_unwinding(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    const int sequential_size = size % 8;
    const int unwinding_size = size - sequential_size;

    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    for (int j = 0, ij = 0; j < size; ++j) {
        int i;
        for (i = 0; i < unwinding_size; i += 8, ij += 8) {
            y_vect[i] += a_matr[ij] * x_vect[j];
            y_vect[i + 1] += a_matr[ij + 1] * x_vect[j];
            y_vect[i + 2] += a_matr[ij + 2] * x_vect[j];
            y_vect[i + 3] += a_matr[ij + 3] * x_vect[j];
            y_vect[i + 4] += a_matr[ij + 4] * x_vect[j];
            y_vect[i + 5] += a_matr[ij + 5] * x_vect[j];
            y_vect[i + 6] += a_matr[ij + 6] * x_vect[j];
            y_vect[i + 7] += a_matr[ij + 7] * x_vect[j];
        }
        for (; i < size; i++) {
            y_vect[i] += a_matr[i + j * size] * x_vect[j];
        }
    }
}

void matvec_sse2(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    __m128d ry0, ry1, ry2, ry3;
    __m128d ra0, ra1, ra2, ra3;
    __m128d rx0;

    const double* ptr_a = a_matr;
    const double* ptr_x = nullptr;

    double* ptr_y = y_vect;

    for (int i = 0; i < size; i += 8) {
        ry0 = ry1 = ry2 = ry3 = _mm_setzero_pd();

        ptr_x = x_vect;

        for (int j = 0; j < size; ++j) {
            _mm_prefetch(static_cast<const void*>(ptr_a + 8), _MM_HINT_NTA);
            if (j % 8 == 0) {
                _mm_prefetch(static_cast<const void*>(ptr_x + 8), _MM_HINT_T0);
            }

            rx0 = _mm_load1_pd(ptr_x);

            ra0 = _mm_load_pd(ptr_a);
            ra1 = _mm_load_pd(ptr_a + 2);
            ra2 = _mm_load_pd(ptr_a + 4);
            ra3 = _mm_load_pd(ptr_a + 6);

            ptr_a += 8;
            ptr_x++;

            ra0 = _mm_mul_pd(ra0, rx0);
            ra1 = _mm_mul_pd(ra1, rx0);
            ra2 = _mm_mul_pd(ra2, rx0);
            ra3 = _mm_mul_pd(ra3, rx0);

            ry0 = _mm_add_pd(ry0, ra0);
            ry1 = _mm_add_pd(ry1, ra1);
            ry2 = _mm_add_pd(ry2, ra2);
            ry3 = _mm_add_pd(ry3, ra3);
        }

        _mm_store_pd(ptr_y, ry0);
        _mm_store_pd(ptr_y + 2, ry1);
        _mm_store_pd(ptr_y + 4, ry2);
        _mm_store_pd(ptr_y + 6, ry3);

        ptr_y += 8;
    }
}

void matvec_sse2_unwinding(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    __m128d ry0, ry1, ry2, ry3;
    __m128d ra0, ra1, ra2, ra3;
    __m128d rx0;

    const double* ptr_a = a_matr;
    const double* ptr_x;
    double* ptr_y;

    const int outer_unwinding = 8;
    const int inner_unwinding = 4;

    for (int i = 0; i < size; i += outer_unwinding) {
        ry0 = ry1 = ry2 = ry3 = _mm_setzero_pd();

        ptr_y = y_vect + i;
        ptr_x = x_vect;

        for (int j = 0; j < size; j += inner_unwinding) {
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 1)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 2)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 3)), _MM_HINT_NTA);

            if (j % inner_unwinding == 0) {
                _mm_prefetch(static_cast<const void*>(ptr_x + inner_unwinding), _MM_HINT_T0);
            }

            // unwinding #0
            rx0 = _mm_load1_pd(ptr_x);

            ra0 = _mm_load_pd(ptr_a);
            ra1 = _mm_load_pd(ptr_a + 2);
            ra2 = _mm_load_pd(ptr_a + 4);
            ra3 = _mm_load_pd(ptr_a + 6);

            ra0 = _mm_mul_pd(ra0, rx0);
            ra1 = _mm_mul_pd(ra1, rx0);
            ra2 = _mm_mul_pd(ra2, rx0);
            ra3 = _mm_mul_pd(ra3, rx0);

            ry0 = _mm_add_pd(ry0, ra0);
            ry1 = _mm_add_pd(ry1, ra1);
            ry2 = _mm_add_pd(ry2, ra2);
            ry3 = _mm_add_pd(ry3, ra3);

            // unwinding #1
            rx0 = _mm_load1_pd(ptr_x + 1);

            ra0 = _mm_load_pd(ptr_a + 8);
            ra1 = _mm_load_pd(ptr_a + 10);
            ra2 = _mm_load_pd(ptr_a + 12);
            ra3 = _mm_load_pd(ptr_a + 14);

            ra0 = _mm_mul_pd(ra0, rx0);
            ra1 = _mm_mul_pd(ra1, rx0);
            ra2 = _mm_mul_pd(ra2, rx0);
            ra3 = _mm_mul_pd(ra3, rx0);

            ry0 = _mm_add_pd(ry0, ra0);
            ry1 = _mm_add_pd(ry1, ra1);
            ry2 = _mm_add_pd(ry2, ra2);
            ry3 = _mm_add_pd(ry3, ra3);

            // unwinding #2
            rx0 = _mm_load1_pd(ptr_x + 2);

            ra0 = _mm_load_pd(ptr_a + 16);
            ra1 = _mm_load_pd(ptr_a + 18);
            ra2 = _mm_load_pd(ptr_a + 20);
            ra3 = _mm_load_pd(ptr_a + 22);

            ra0 = _mm_mul_pd(ra0, rx0);
            ra1 = _mm_mul_pd(ra1, rx0);
            ra2 = _mm_mul_pd(ra2, rx0);
            ra3 = _mm_mul_pd(ra3, rx0);

            ry0 = _mm_add_pd(ry0, ra0);
            ry1 = _mm_add_pd(ry1, ra1);
            ry2 = _mm_add_pd(ry2, ra2);
            ry3 = _mm_add_pd(ry3, ra3);

            // unwinding #3
            rx0 = _mm_load1_pd(ptr_x + 3);

            ra0 = _mm_load_pd(ptr_a + 24);
            ra1 = _mm_load_pd(ptr_a + 26);
            ra2 = _mm_load_pd(ptr_a + 28);
            ra3 = _mm_load_pd(ptr_a + 30);

            ra0 = _mm_mul_pd(ra0, rx0);
            ra1 = _mm_mul_pd(ra1, rx0);
            ra2 = _mm_mul_pd(ra2, rx0);
            ra3 = _mm_mul_pd(ra3, rx0);

            ry0 = _mm_add_pd(ry0, ra0);
            ry1 = _mm_add_pd(ry1, ra1);
            ry2 = _mm_add_pd(ry2, ra2);
            ry3 = _mm_add_pd(ry3, ra3);

            ptr_a += inner_unwinding * outer_unwinding;
            ptr_x += inner_unwinding;
        }

        _mm_store_pd(ptr_y, ry0);
        _mm_store_pd(ptr_y + 2, ry1);
        _mm_store_pd(ptr_y + 4, ry2);
        _mm_store_pd(ptr_y + 6, ry3);
    }
}

void matvec_avx(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    __m256d ry0, ry1, ry2, ry3;
    __m256d ra0, ra1, ra2, ra3;
    __m256d rx0, rx1;

    const double* ptr_a = a_matr;
    const double* ptr_x;

    double* ptr_y = y_vect;

    const int outer_unwinding = 8;
    const int inner_unwinding = 8;

    for (int i = 0; i < size; i += outer_unwinding) {
        ry0 = _mm256_setzero_pd();
        ry1 = ry2 = ry3 = ry0;

        ptr_x = x_vect;

        for (int j = 0; j < size; j += inner_unwinding) {
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 1)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 2)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 3)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 4)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 5)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 6)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 7)), _MM_HINT_NTA);

            _mm_prefetch(static_cast<const void*>(ptr_x + inner_unwinding), _MM_HINT_T0);

            // unwinding #0
            rx0 = _mm256_broadcast_sd(ptr_x);
            rx1 = _mm256_broadcast_sd(ptr_x + 1);

            ra0 = _mm256_load_pd(ptr_a);
            ra1 = _mm256_load_pd(ptr_a + 4);
            ra2 = _mm256_load_pd(ptr_a + outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + outer_unwinding + 4);

            ra0 = _mm256_mul_pd(ra0, rx0);
            ra1 = _mm256_mul_pd(ra1, rx0);
            ra2 = _mm256_mul_pd(ra2, rx1);
            ra3 = _mm256_mul_pd(ra3, rx1);

            ry0 = _mm256_add_pd(ry0, ra0);
            ry1 = _mm256_add_pd(ry1, ra1);
            ry2 = _mm256_add_pd(ry2, ra2);
            ry3 = _mm256_add_pd(ry3, ra3);

            // unwinding #1
            rx0 = _mm256_broadcast_sd(ptr_x + 2);
            rx1 = _mm256_broadcast_sd(ptr_x + 3);

            ra0 = _mm256_load_pd(ptr_a + 2 * outer_unwinding);
            ra1 = _mm256_load_pd(ptr_a + 2 * outer_unwinding + 4);
            ra2 = _mm256_load_pd(ptr_a + 3 * outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + 3 * outer_unwinding + 4);

            ra0 = _mm256_mul_pd(ra0, rx0);
            ra1 = _mm256_mul_pd(ra1, rx0);
            ra2 = _mm256_mul_pd(ra2, rx1);
            ra3 = _mm256_mul_pd(ra3, rx1);

            ry0 = _mm256_add_pd(ry0, ra0);
            ry1 = _mm256_add_pd(ry1, ra1);
            ry2 = _mm256_add_pd(ry2, ra2);
            ry3 = _mm256_add_pd(ry3, ra3);

            // unwinding #2
            rx0 = _mm256_broadcast_sd(ptr_x + 4);
            rx1 = _mm256_broadcast_sd(ptr_x + 5);

            ra0 = _mm256_load_pd(ptr_a + 4 * outer_unwinding);
            ra1 = _mm256_load_pd(ptr_a + 4 * outer_unwinding + 4);
            ra2 = _mm256_load_pd(ptr_a + 5 * outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + 5 * outer_unwinding + 4);

            ra0 = _mm256_mul_pd(ra0, rx0);
            ra1 = _mm256_mul_pd(ra1, rx0);
            ra2 = _mm256_mul_pd(ra2, rx1);
            ra3 = _mm256_mul_pd(ra3, rx1);

            ry0 = _mm256_add_pd(ry0, ra0);
            ry1 = _mm256_add_pd(ry1, ra1);
            ry2 = _mm256_add_pd(ry2, ra2);
            ry3 = _mm256_add_pd(ry3, ra3);

            // unwinding #3
            rx0 = _mm256_broadcast_sd(ptr_x + 6);
            rx1 = _mm256_broadcast_sd(ptr_x + 7);

            ra0 = _mm256_load_pd(ptr_a + 6 * outer_unwinding);
            ra1 = _mm256_load_pd(ptr_a + 6 * outer_unwinding + 4);
            ra2 = _mm256_load_pd(ptr_a + 7 * outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + 7 * outer_unwinding + 4);

            ra0 = _mm256_mul_pd(ra0, rx0);
            ra1 = _mm256_mul_pd(ra1, rx0);
            ra2 = _mm256_mul_pd(ra2, rx1);
            ra3 = _mm256_mul_pd(ra3, rx1);

            ry0 = _mm256_add_pd(ry0, ra0);
            ry1 = _mm256_add_pd(ry1, ra1);
            ry2 = _mm256_add_pd(ry2, ra2);
            ry3 = _mm256_add_pd(ry3, ra3);

            ptr_a += outer_unwinding * inner_unwinding;
            ptr_x += inner_unwinding;
        }

        ry0 = _mm256_add_pd(ry0, ry2);
        ry1 = _mm256_add_pd(ry1, ry3);

        _mm256_store_pd(ptr_y, ry0);
        _mm256_store_pd(ptr_y + 4, ry1);

        ptr_y += outer_unwinding;
    }
}

void matvec_fma_avx(double* y_vect, const double* a_matr, const double* x_vect, int size)
{
    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));

    __m256d ry0, ry1, ry2, ry3;
    __m256d ra0, ra1, ra2, ra3;
    __m256d rx0, rx1;

    const double* ptr_a = a_matr;
    const double* ptr_x;

    double* ptr_y;

    const int outer_unwinding = 8;
    const int inner_unwinding = 8;

    for (int i = 0; i < size; i += outer_unwinding) {
        ry0 = ry1 = ry2 = ry3 = _mm256_setzero_pd();

        ptr_y = y_vect + i;
        ptr_x = x_vect;

        for (int j = 0; j < size; j += inner_unwinding) {
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * inner_unwinding), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 1)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 2)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 3)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 4)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 5)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 6)), _MM_HINT_NTA);
            _mm_prefetch(static_cast<const void*>(ptr_a + outer_unwinding * (inner_unwinding + 7)), _MM_HINT_NTA);

            _mm_prefetch(static_cast<const void*>(ptr_x + inner_unwinding), _MM_HINT_T0);

            // unwinding #0
            rx0 = _mm256_broadcast_sd(ptr_x);
            rx1 = _mm256_broadcast_sd(ptr_x + 1);

            ra0 = _mm256_load_pd(ptr_a);
            ra1 = _mm256_load_pd(ptr_a + 4);
            ra2 = _mm256_load_pd(ptr_a + outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + outer_unwinding + 4);

            ry0 = _mm256_fmadd_pd(ra0, rx0, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx0, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx1, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx1, ry3);

            // unwinding #1
            rx0 = _mm256_broadcast_sd(ptr_x + 2);
            rx1 = _mm256_broadcast_sd(ptr_x + 3);

            ra0 = _mm256_load_pd(ptr_a + 2 * outer_unwinding);
            ra1 = _mm256_load_pd(ptr_a + 2 * outer_unwinding + 4);
            ra2 = _mm256_load_pd(ptr_a + 3 * outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + 3 * outer_unwinding + 4);

            ry0 = _mm256_fmadd_pd(ra0, rx0, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx0, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx1, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx1, ry3);

            // unwinding #2
            rx0 = _mm256_broadcast_sd(ptr_x + 4);
            rx1 = _mm256_broadcast_sd(ptr_x + 5);

            ra0 = _mm256_load_pd(ptr_a + 4 * outer_unwinding);
            ra1 = _mm256_load_pd(ptr_a + 4 * outer_unwinding + 4);
            ra2 = _mm256_load_pd(ptr_a + 5 * outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + 5 * outer_unwinding + 4);

            ry0 = _mm256_fmadd_pd(ra0, rx0, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx0, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx1, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx1, ry3);

            // unwinding #3
            rx0 = _mm256_broadcast_sd(ptr_x + 6);
            rx1 = _mm256_broadcast_sd(ptr_x + 7);

            ra0 = _mm256_load_pd(ptr_a + 6 * outer_unwinding);
            ra1 = _mm256_load_pd(ptr_a + 6 * outer_unwinding + 4);
            ra2 = _mm256_load_pd(ptr_a + 7 * outer_unwinding);
            ra3 = _mm256_load_pd(ptr_a + 7 * outer_unwinding + 4);

            ry0 = _mm256_fmadd_pd(ra0, rx0, ry0);
            ry1 = _mm256_fmadd_pd(ra1, rx0, ry1);
            ry2 = _mm256_fmadd_pd(ra2, rx1, ry2);
            ry3 = _mm256_fmadd_pd(ra3, rx1, ry3);

            ptr_a += outer_unwinding * inner_unwinding;
            ptr_x += inner_unwinding;
        }

        ry0 = _mm256_add_pd(ry0, ry2);
        ry1 = _mm256_add_pd(ry1, ry3);

        _mm256_store_pd(ptr_y, ry0);
        _mm256_store_pd(ptr_y + 4, ry1);
    }
}
