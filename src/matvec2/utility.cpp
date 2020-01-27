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
    for (int i = 0, ij = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j, ++ij) {
            a_matr[ij] = (i == j) ? 10.0 : (i + 1.0);
        }

        x_vect[i] = 1.0;
    }
}
