#include "dot_data_t.hpp"
#include "dot_prod_funcs.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using namespace std::chrono;

class args_t {
private:
    const int _threads;
    const int _repetitions;
    const int _size;

public:
    args_t(int threads, int repetitions, int size)
        : _threads(threads)
        , _repetitions(repetitions)
        , _size(size)
    {
    }

    args_t(const args_t& obj)
        : _threads(obj.threads())
        , _repetitions(obj.repetitions())
        , _size(obj.size())
    {
    }

    inline const int& threads() const { return _threads; }
    inline const int& repetitions() const { return _repetitions; }
    inline const int& size() const { return _size; }
};

static void benchmark(int threads_count, int repetitions, int size);
static vector<args_t> parseargs(int argc, char* argv[]);

inline long instant() { return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(); }

int main(int argc, char* argv[])
{
    for (const args_t& param : parseargs(argc, argv)) {
        benchmark(param.threads(), param.repetitions(), param.size());
    }
    return 0;
}

static void benchmark(int threads_count, int repetitions, int size)
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
        x_vect[i] = y_vect[i] = i % 100;
    }

    vector<dot_data_t> datas;
    datas.reserve(threads_count);
    for (int i = 0; i < threads_count; ++i) {
        datas.emplace_back(dot_data_t(x_vect, y_vect, repetitions, size, i * slice_size, slice_size));
    }

    vector<thread> threads;
    threads.reserve(threads_count);

    // naive algorithm benchmark

    long start = instant();
    for (int i = 0; i < threads_count; ++i) {
        threads.emplace_back(thread(&dot_prod_naive, &datas[i]));
    }

    for (thread& thread : threads) {
        thread.join();
    }

    double naive_dot_prod = 0.0;
    for (dot_data_t& data : datas) {
        naive_dot_prod += data.result();
    }

    long stop = instant();
    cout << "[naive]   " << static_cast<double>(stop - start) / 1000.0 << "ms" << endl;
    threads.clear();

    // SSE2 algorithm benchmark

    start = instant();
    for (int i = 0; i < threads_count; ++i) {
        threads.emplace_back(thread(&dot_prod_sse2, &datas[i]));
    }

    for (thread& thread : threads) {
        thread.join();
    }

    double dot_prod = 0.0;
    for (dot_data_t& data : datas) {
        dot_prod += data.result();
    }
    bool correct = abs(dot_prod - naive_dot_prod) < 1.0e-9;

    stop = instant();
    cout << "[SSE2]    " << static_cast<double>(stop - start) / 1000.0 << "ms [" << (correct ? "OK" : "ERROR") << "]"
         << endl;
    threads.clear();

    // AVX algorithm benchmark

    for (int i = 0; i < threads_count; ++i) {
        threads.emplace_back(thread(&dot_prod_avx, &datas[i]));
    }

    for (thread& thread : threads) {
        thread.join();
    }

    dot_prod = 0.0;
    for (dot_data_t& data : datas) {
        dot_prod += data.result();
    }
    correct = abs(dot_prod - naive_dot_prod) < 1.0e-9;

    stop = instant();
    cout << "[AVX]     " << static_cast<double>(stop - start) / 1000.0 << "ms [" << (correct ? "OK" : "ERROR") << "]"
         << endl;
    threads.clear();

    // AVX+FMA algorithm benchmark

    for (int i = 0; i < threads_count; ++i) {
        threads.emplace_back(thread(&dot_prod_avx_fma, &datas[i]));
    }

    for (thread& thread : threads) {
        thread.join();
    }

    dot_prod = 0.0;
    for (dot_data_t& data : datas) {
        dot_prod += data.result();
    }
    correct = abs(dot_prod - naive_dot_prod) < 1.0e-9;

    stop = instant();
    cout << "[AVX+FMA] " << static_cast<double>(stop - start) / 1000.0 << "ms [" << (correct ? "OK" : "ERROR") << "]"
         << endl;
    threads.clear();

    free(x_vect);
    free(y_vect);
}

static vector<args_t> parseargs(int argc, char* argv[])
{
    vector<args_t> params;
    if (argc != 4) {
        cerr << "[ERROR] expected call " << argv[0] << " <threads> <repetitions> <size>" << endl;
        return vector<args_t>();
    }

    int threads = 0;
    int repetitions = 0;
    int size = 0;

    try {
        threads = stoi(argv[1]);
    } catch (invalid_argument& e) {
        cerr << "[ERROR] " << argv[1] << " is not a valid integer" << endl;
        exit(1);
    }

    try {
        repetitions = stoi(argv[2]);
    } catch (invalid_argument& e) {
        cerr << "[ERROR] " << argv[2] << " is not a valid integer" << endl;
        exit(1);
    }

    try {
        size = stoi(argv[3]);
    } catch (invalid_argument& e) {
        cerr << "[ERROR] " << argv[3] << " is not a valid integer" << endl;
        exit(1);
    }

    params.emplace_back(args_t(threads, repetitions, size));

    if (params.empty()) {
        cerr << "[ERROR] no vector size provided" << endl;
    }
    return params;
}
