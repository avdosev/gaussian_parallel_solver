#include <iostream>
#include "gaus.h"

void print(std::ostream &s, Matrix2D<double> matrix) {
    s << '[';
    for (size_t i = 0; i < matrix.n1; i++) {
        s << '[' << ' ';
        for (size_t j = 0; j < matrix.n2; j++) {
            s << matrix.get(i, j) << ' ';
        }
        s << ']';
        if (i + 1 != matrix.n1)
            s << std::endl;
    }
    s << ']' << std::endl;
}

void example() {
    Matrix2D<double> task{3, 4};
    task.data = std::vector<double>{{
                                            2, 1, -1, 8,
                                            -3, -1, 2, -11,
                                            -2, 1, 2, -3
                                    }};
    print(std::cout, task);
    auto result = gauss_solve(task);
    print(std::cout, result);
}

int main() {
    example();
    return 0;
}
