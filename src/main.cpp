#include <iostream>
#include <vector>
#include <random>
#include "gaus.h"

void print(std::ostream &s, const Matrix2D<double>& matrix) {
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

void print(std::ostream &s, const std::vector<double>& arr) {
    s << '[' << ' ';
        for (auto item : arr) {
            s << item << ' ';
        }
    s << ']' << std::endl;
}

std::pair<Matrix2D<double>, std::vector<double>> generate_matrix(size_t n) {
    std::mt19937 gen{69};
    Matrix2D<double> m{n, n+1};
    std::vector<double> solve(n);
    std::normal_distribution<> d{-10,10};

    for (auto& item: solve) {
        item = d(gen);
    }

    for (size_t i = 0; i < n; i++) {
        double sum = 0;
        for (size_t j = 0; j < n; j++) {
            auto item = d(gen);
            m.get(i, j) = item;
            sum += item * solve[j];
        }
        m.get(i, n) = sum;
    }

    return {m, solve};
}

bool equal(std::vector<double>& arr1, std::vector<double>& arr2, double eps = 0.0001) {
    if (arr1.size() != arr2.size()) return false;
    for (size_t i = 0; i < arr1.size(); i++) {
        if (std::abs(arr1[i] - arr2[i]) > eps) return false;
    }
    return true;
}

void example() {
    auto [task, solve] = generate_matrix(3);
    print(std::cout, solve);
    print(std::cout, task);
    auto result = gauss_solve_omp(task);
    print(std::cout, result);
    std::cout << "equal: " << equal(solve, result.data) << std::endl;
}

int main() {
    example();
    return 0;
}
