#ifndef __DOT_PROD_DOT_DATA_T__
#define __DOT_PROD_DOT_DATA_T__

class dot_data_t {
private:
    double* const _x_vect;
    double* const _y_vect;
    const int _repetitions;
    const int _size;
    const int _sub_offset;
    const int _sub_size;
    double _result;
    int _status;

public:
    dot_data_t(double* const x_vect, double* const y_vect, int repetitions, int size, int sub_offset, int sub_size)
        : _x_vect(x_vect)
        , _y_vect(y_vect)
        , _repetitions(repetitions)
        , _size(size)
        , _sub_offset(sub_offset)
        , _sub_size(sub_size)
        , _result(0.0)
        , _status(0)
    {
    }

    dot_data_t(const dot_data_t& obj)
        : _x_vect(obj.x_vect())
        , _y_vect(obj.y_vect())
        , _repetitions(obj.repetitions())
        , _size(obj.size())
        , _sub_offset(obj.sub_offset())
        , _sub_size(obj.sub_size())
        , _result(obj.result())
        , _status(obj.status())
    {
    }

    inline double* x_vect() const { return _x_vect; }

    inline double* y_vect() const { return _y_vect; }

    inline const int& repetitions() const { return _repetitions; }

    inline const int& size() const { return _size; }

    inline const int& sub_offset() const { return _sub_offset; }

    inline const int& sub_size() const { return _sub_size; }

    inline double result() const { return _result; }

    inline int status() const { return _status; }

    inline void result(double result) { _result = result; }
};

#endif
