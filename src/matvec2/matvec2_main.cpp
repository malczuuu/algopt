#include "matvec2_algorithms.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::chrono;

static void benchmark(int size);
static vector<int> parseargs(int argc, char* argv[]);
static void prepare(double* a_matr, double* x_vect, int size);

inline long instant() { return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(); }

int main(int argc, char* argv[])
{
    for (int param : parseargs(argc, argv)) {
        benchmark(param);
    }

    return 0;
}

static bool check(double* y_vect, double* z_vect, int size)
{
    for (int i = 0; i < size; ++i) {
        if (abs(z_vect[i] - y_vect[i]) > 1.0e-9) {
            return false;
        }
    }

    return true;
}

static void benchmark(int size)
{
    double* a_matr = nullptr;
    double* x_vect = nullptr;
    double* y_vect = nullptr;
    double* z_vect = nullptr;
    const int block_size = 16;

    size = size / block_size;
    size = block_size * size;

    try {
        a_matr = static_cast<double*>(aligned_alloc(32, size * size * sizeof(double)));
        x_vect = static_cast<double*>(aligned_alloc(32, size * sizeof(double)));
        y_vect = static_cast<double*>(aligned_alloc(32, size * sizeof(double)));
        if (a_matr == nullptr || x_vect == nullptr || y_vect == nullptr) {
            throw bad_alloc();
        }
        z_vect = new double[size];
    } catch (bad_alloc&) {
        cerr << "[ERROR] memory allocation failed" << endl;
        exit(1);
    }
    memset(static_cast<void*>(a_matr), 0, size * size * sizeof(double));
    memset(static_cast<void*>(x_vect), 0, size * sizeof(double));
    memset(static_cast<void*>(y_vect), 0, size * sizeof(double));
    memset(static_cast<void*>(z_vect), 0, size * sizeof(double));

    prepare(a_matr, x_vect, size);

    cout << "[BENCHMARK] (size=" << size << ")" << endl;

    long start = instant();
    matvec2_naive(z_vect, a_matr, x_vect, size);
    cout << "[BENCHMARK] naive algorithm (time=" << (static_cast<double>(instant() - start) / 1000.0) << "s)" << endl;

    start = instant();
    matvec2_sse2(y_vect, a_matr, x_vect, size);
    bool correct = check(y_vect, z_vect, size);
    cout << "[BENCHMARK] SSE2 (time=" << (static_cast<double>(instant() - start) / 1000.0) << "s) ["
         << (correct ? "OK" : "ERROR") << "]" << endl;

    start = instant();
    matvec2_avx(y_vect, a_matr, x_vect, size);
    correct = check(y_vect, z_vect, size);
    cout << "[BENCHMARK] AVX (time=" << (static_cast<double>(instant() - start) / 1000.0) << "s) ["
         << (correct ? "OK" : "ERROR") << "]" << endl;

    start = instant();
    matvec2_fma_avx(y_vect, a_matr, x_vect, size);
    correct = check(y_vect, z_vect, size);
    cout << "[BENCHMARK] FMA + AVX (time=" << (static_cast<double>(instant() - start) / 1000.0) << "s) ["
         << (correct ? "OK" : "ERROR") << "]" << endl;

    free(a_matr);
    free(x_vect);
    free(y_vect);
    delete[] z_vect;
}

static vector<int> parseargs(int argc, char* argv[])
{
    vector<int> params;
    for (int i = 1; i < argc; ++i) {
        try {
            params.emplace_back(stoi(argv[i]));
        } catch (invalid_argument&) {
            cerr << "[ERROR] " << argv[i] << " is not a valid argument" << endl;
            exit(1);
        }
    }
    if (params.empty()) {
        cerr << "[ERROR] no vector size provided" << endl;
    }
    return params;
}

static void prepare(double* a_matr, double* x_vect, int size)
{
    for (int i = 0, ij = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j, ++ij) {
            a_matr[ij] = (i == j) ? 10.0 : (i + 1.0);
        }

        x_vect[i] = 1.0;
    }
}
