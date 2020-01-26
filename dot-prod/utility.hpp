#ifndef __DOT_PROD_UTILITY__
#define __DOT_PROD_UTILITY__

#include <vector>

class args_t {
private:
    const int _threads;
    const int _repetitions;
    const int _size;

public:
    args_t(const int threads, const int repetitions, const int size)
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

    inline const int threads() const { return _threads; }
    inline const int repetitions() const { return _repetitions; }
    inline const int size() const { return _size; }
};

std::vector<args_t> parseargs(int argc, char* argv[]);

#endif
