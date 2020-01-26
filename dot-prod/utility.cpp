#include "utility.hpp"
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace chrono;

namespace dotprod {

vector<args_t> parseargs(int argc, char* argv[])
{
    vector<args_t> params;
    if (argc != 4) {
        cerr << "[ERROR] expected call" << argv[0] << " <threads> <repetitions> <size>" << endl;
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

long instant() { return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(); }
}