#include "dot_data_t.hpp"
#include "thread_funcs.hpp"
#include "utility.hpp"
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using namespace dotprod;

void benchmark(int threads_count, int repetitions, int size);

int main(int argc, char* argv[])
{
    for (args_t param : parseargs(argc, argv)) {
        benchmark(param.threads(), param.repetitions(), param.size());
    }
    return 0;
}

void benchmark(int threads_count, int repetitions, int size)
{
    int slice_size = size / threads_count;
    size = slice_size * threads_count;

    cout << "[BENCHNARK] (threads_count=" << threads_count << ", repetitions=" << repetitions << ", size=" << size
         << ")" << endl;

    double* x_vect = nullptr;
    double* y_vect = nullptr;

    try {
        x_vect = static_cast<double*>(aligned_alloc(32, size * sizeof(double)));
        y_vect = static_cast<double*>(aligned_alloc(32, size * sizeof(double)));
        if (x_vect == nullptr || y_vect == nullptr) {
            throw bad_alloc();
        }
    } catch (bad_alloc&) {
        cerr << "[ERROR] memory allocation failed" << endl;
        exit(1);
    }

    for (int i = 0; i < size; ++i) {
        x_vect[i] = y_vect[i] = 1.0;
    }

    vector<dot_data_t> datas;
    for (int i = 0; i < threads_count; ++i) {
        datas.emplace_back(dot_data_t(x_vect, y_vect, repetitions, size, i * slice_size, slice_size));
    }

    vector<thread> threads;
    for (int i = 0; i < threads_count; ++i) {
        threads.emplace_back(thread(&dot_prod_naive, &datas[i]));
    }

    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }

    double dot_prod = 0.0;
    for (int i = 0; i < datas.size(); ++i) {
        dot_prod += datas[i].result();
    }

    cout << dot_prod << endl;

    free(x_vect);
    free(y_vect);
}
