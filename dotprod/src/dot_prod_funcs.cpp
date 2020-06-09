#include "dot_prod_funcs.hpp"
#include <emmintrin.h>
#include <immintrin.h>

void dot_prod_naive(dot_data_t* data)
{
    double result = 0.0;
    for (int rep = 0; rep < data->repetitions(); ++rep) {
        for (int i = 0; i < data->sub_size(); ++i) {
            double x = data->x_vect()[data->sub_offset() + i];
            double y = data->y_vect()[data->sub_offset() + i];
            result += x * y;
        }
    }
    data->result(result);
}

void dot_prod_sse2(dot_data_t* data) { data->result(-1.0); }

void dot_prod_avx(dot_data_t* data) { data->result(-1.0); }

void dot_prod_avx_fma(dot_data_t* data)
{
    __m256d sum0, sum1, sum2, sum3;
    __m256d rx0, rx1, rx2, rx3;
    __m256d ry0, ry1, ry2, ry3;

    double* buf0 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));
    double* buf1 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));
    double* buf2 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));
    double* buf3 = static_cast<double*>(aligned_alloc(32, 4 * sizeof(double)));

    const int loop_unwinding = 16;

    const int rest = data->sub_size() % loop_unwinding;
    const int mainSize = data->sub_size() - rest;

    double subdeterminant = 0.0;

    for (int i = 0; i < data->repetitions(); ++i) {
        double* ptr_x = data->x_vect() + data->sub_offset();
        double* ptr_y = data->y_vect() + data->sub_offset();

        sum0 = sum1 = sum2 = sum3 = _mm256_setzero_pd();

        for (int j = 0; j < mainSize; j += loop_unwinding) {
            _mm_prefetch((const char*)(ptr_x + loop_unwinding), _MM_HINT_T0);
            _mm_prefetch((const char*)(ptr_x + loop_unwinding + 8), _MM_HINT_T0);
            _mm_prefetch((const char*)(ptr_y + loop_unwinding), _MM_HINT_T0);
            _mm_prefetch((const char*)(ptr_y + loop_unwinding + 8), _MM_HINT_T0);

            rx0 = _mm256_loadu_pd(ptr_x);
            rx1 = _mm256_loadu_pd(ptr_x + 4);
            rx2 = _mm256_loadu_pd(ptr_x + 8);
            rx3 = _mm256_loadu_pd(ptr_x + 12);

            ry0 = _mm256_loadu_pd(ptr_y);
            ry1 = _mm256_loadu_pd(ptr_y + 4);
            ry2 = _mm256_loadu_pd(ptr_y + 8);
            ry3 = _mm256_loadu_pd(ptr_y + 12);

            sum0 = _mm256_fmadd_pd(rx0, ry0, sum0);
            sum1 = _mm256_fmadd_pd(rx1, ry1, sum1);
            sum2 = _mm256_fmadd_pd(rx2, ry2, sum2);
            sum3 = _mm256_fmadd_pd(rx3, ry3, sum3);

            ptr_x += loop_unwinding;
            ptr_y += loop_unwinding;
        }

        _mm256_store_pd(buf0, sum0);
        _mm256_store_pd(buf1, sum1);
        _mm256_store_pd(buf2, sum2);
        _mm256_store_pd(buf3, sum3);

        subdeterminant += buf0[0] + buf0[1] + buf0[2] + buf0[3];
        subdeterminant += buf1[0] + buf1[1] + buf1[2] + buf1[3];
        subdeterminant += buf2[0] + buf2[1] + buf2[2] + buf2[3];
        subdeterminant += buf3[0] + buf3[1] + buf3[2] + buf3[3];

        for (int j = 0; j < rest; ++j) {
            subdeterminant += ptr_x[j] * ptr_y[j];
        }
    }

    data->result(subdeterminant);

    free(buf0);
    free(buf1);
    free(buf2);
    free(buf3);
}
