#include <iostream>
#include <cstring>
#include <vector>
#include "algorithms.hpp"
#include "utility.hpp"

using namespace std;

static void benchmark(int size);

int main(int argc, char *argv[])
{
    for (int param : parseargs(argc, argv))
    {
        benchmark(param);
    }

    return 0;
}

static bool check(double *y_vect, double *z_vect, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (abs(z_vect[i] - y_vect[i]) > 1.0e-9)
        {
            return false;
        }
    }

    return true;
}

static void benchmark(int size)
{
    double *a_matr = nullptr;
    double *x_vect = nullptr;
    double *y_vect = nullptr;
    double *z_vect = nullptr;
    const int block_size = 8;

    size = size / block_size;
    size = block_size * size;

    try
    {
        a_matr = static_cast<double *>(aligned_alloc(32, size * size * sizeof(double)));
        x_vect = static_cast<double *>(aligned_alloc(32, size * sizeof(double)));
        y_vect = static_cast<double *>(aligned_alloc(32, size * sizeof(double)));
        if (a_matr == nullptr || x_vect == nullptr || y_vect == nullptr)
        {
            throw bad_alloc();
        }
        z_vect = new double[size];
    }
    catch (bad_alloc &)
    {
        cerr << "[ERROR] memory allocation failed" << endl;
        exit(1);
    }
    memset(static_cast<void *>(a_matr), 0, size * size * sizeof(double));
    memset(static_cast<void *>(x_vect), 0, size * sizeof(double));
    memset(static_cast<void *>(y_vect), 0, size * sizeof(double));

    prepare(a_matr, x_vect, size);

    cout << "[BENCHMARK] (size=" << size << ", block_size=" << block_size << ")" << endl;


    cout << "[START] naive algorithm" << endl;
    long start = instant();
    matvec_naive(z_vect, a_matr, x_vect, size);
    cout
            << "[STOP]  naive algorithm (time="
            << (static_cast<double>(instant() - start) / 1000.0) << "s)" << endl;


    cout << "[START] fixed memory jumps" << endl;
    start = instant();
    matvec_fixed_memjumps(y_vect, a_matr, x_vect, size);
    bool correct = check(y_vect, z_vect, size);
    cout
            << "[STOP]  fixed memory jumps (time="
            << (static_cast<double>(instant() - start) / 1000.0) << "s) "
            << (correct ? "OK" : "ERROR") << endl;


    cout << "[START] loop unwinding" << endl;
    start = instant();
    matvec_unwinding(y_vect, a_matr, x_vect, size);
    correct = check(y_vect, z_vect, size);
    cout
            << "[STOP]  loop unwinding (time="
            << (static_cast<double>(instant() - start) / 1000.0) << "s) "
            << (correct ? "OK" : "ERROR") << endl;


    cout << "[START] SSE2" << endl;

    prepare_block(a_matr, x_vect, size, block_size);

    start = instant();
    matvec_sse2(y_vect, a_matr, x_vect, size);
    correct = check(y_vect, z_vect, size);
    cout
            << "[STOP]  SSE2 (time="
            << (static_cast<double>(instant() - start) / 1000.0) << "s) "
            << (correct ? "OK" : "ERROR") << endl;


    cout << "[START] AVX" << endl;
    start = instant();
    matvec_avx(y_vect, a_matr, x_vect, size);
    correct = check(y_vect, z_vect, size);
    cout
            << "[STOP]  AVX (time="
            << (static_cast<double>(instant() - start) / 1000.0) << "s) "
            << (correct ? "OK" : "ERROR") << endl;


    free(a_matr);
    free(x_vect);
    free(y_vect);
    delete[] z_vect;
}
