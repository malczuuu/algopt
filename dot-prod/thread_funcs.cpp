#include "thread_funcs.hpp"

namespace dotprod {

void dot_prod_naive(dot_data_t* data)
{
    for (int rep = 0; rep < data->repetitions(); ++rep) {
        for (int i = 0; i < data->sub_size(); ++i) {
            double x = data->x_vect()[data->sub_offset() + i];
            double y = data->y_vect()[data->sub_offset() + i];
            data->result(data->result() + x * y);
        }
    }
}
}
