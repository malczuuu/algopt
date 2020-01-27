#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::chrono;

vector<int> parseargs(int argc, char* argv[])
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

long instant() { return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(); }

void prepare(double* a_matr, double* x_vect, int size)
{
    for (int j = 0, ij = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i, ++ij) {
            a_matr[ij] = (i == j) ? 10.0 : i + 1.0;
        }
        x_vect[j] = 1.0;
    }
}

void prepare_block(double* a_matr, double* x_vect, int size, int block_size)
{
    int ij = 0;

    for (int ii = 0; ii < size; ii += block_size) {
        for (int j = 0; j < size; ++j) {
            for (int i = ii; i < ii + block_size; ++i) {
                a_matr[ij] = (i == j) ? 10.0 : i + 1.0;
                ++ij;
            }
        }
    }
    for (int j = 0; j < size; ++j) {
        x_vect[j] = 1.0;
    }
}
