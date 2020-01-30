#ifndef __DOT_PROD_THREAD_FUNCS_T__
#define __DOT_PROD_THREAD_FUNCS_T__

#include "dot_data_t.hpp"

void dot_prod_naive(dot_data_t* data);

void dot_prod_sse2(dot_data_t* data);

void dot_prod_avx(dot_data_t* data);

void dot_prod_avx_fma(dot_data_t* data);

#endif
