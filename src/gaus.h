#pragma once

#include <vector>
#include <tuple>

template<typename T = double>
class Matrix2D {
public:
    std::vector<T> data;
    size_t n1, n2;

    Matrix2D() = default;
    Matrix2D(size_t n1, size_t n2) : n1(n1), n2(n2), data(n1*n2, 0) {}

    T& get(size_t i, size_t j) {
        return data[i*n2 + j];
    }

    const T& get(size_t i, size_t j) const {
        return data[i*n2 + j];
    }
};

// Гаус слау
template<typename T>
Matrix2D<T> gauss_solve(Matrix2D<T> matrix) {
    size_t n1 = matrix.n1; // строки
    size_t n2 = matrix.n2; // столбцы

    auto matrix_clone = matrix;

    // Прямой ход (Зануление нижнего левого угла)
    for (size_t k = 0; k < n1; k++) {
        for (size_t i = 0; i < n2; i++) {
            matrix_clone.get(k, i) /= matrix.get(k, k);
        }
        for (size_t i = k + 1; i < n1; i++) {
            double K = matrix_clone.get(i, k) / matrix_clone.get(k, k);
            for (size_t j = 0; j < n2; j++) {
                matrix_clone.get(i, j) -= matrix_clone.get(k, j) * K;
            }
        }
        for (size_t i = 0; i < n1; i++)
            for (size_t j = 0; j < n2; j++)
                matrix.get(i, j) = matrix_clone.get(i, j);
    }

    // Обратный ход (Зануление верхнего правого угла)
    for (size_t k1 = n1; k1 > 0; k1--) {
        auto k = k1 - 1;
        for (size_t i1 = n2 - 1; i1 > 0; i1--) {
            auto i = i1 - 1;
            matrix_clone.get(k, i) = matrix_clone.get(k, i) / matrix.get(k, k);
        }
        for (size_t i1 = k; i1 > 0; i1--) {
            auto i = i1 - 1;
            T K = matrix_clone.get(i, k) / matrix_clone.get(k, k);
            for (size_t j1 = n2; j1 > 0; j1--) {
                auto j = j1 - 1;
                matrix_clone.get(i, j) = matrix_clone.get(i, j) - matrix_clone.get(k, j) * K;
            }
        }
    }

    // Отделяем от общей матрицы ответы
    Matrix2D<T> answer{n1, 1};
    for (size_t i = 0; i < n1; i++)
        answer.get(i, 0) = matrix_clone.get(i, n2-1);

    return answer;
}

template<typename T>
Matrix2D<T> gauss_solve_omp(Matrix2D<T> matrix) {
    size_t n1 = matrix.n1; // строки
    size_t n2 = matrix.n2; // столбцы

    auto matrix_clone = matrix;

    // Прямой ход (Зануление нижнего левого угла)
    for (size_t k = 0; k < n1; k++) {
        auto main_k = matrix.get(k, k);
#pragma omp parallel for
        for (size_t i = 0; i < n2; i++) {
            matrix_clone.get(k, i) /= main_k;
        }
#pragma omp parallel for
        for (size_t i = k + 1; i < n1; i++) {
            double K = matrix_clone.get(i, k) / matrix_clone.get(k, k);
            for (size_t j = 0; j < n2; j++) {
                matrix_clone.get(i, j) -= matrix_clone.get(k, j) * K;
            }
        }
#pragma omp parallel for
        for (size_t i = 0; i < n1; i++)
            for (size_t j = 0; j < n2; j++)
                matrix.get(i, j) = matrix_clone.get(i, j);
    }

    // Обратный ход (Зануление верхнего правого угла)
    for (size_t k1 = n1; k1 > 0; k1--) {
        auto k = k1 - 1;
#pragma omp parallel for
        for (size_t i1 = n2 - 1; i1 > 0; i1--) {
            auto i = i1 - 1;
            matrix_clone.get(k, i) /= matrix.get(k, k);
        }
#pragma omp parallel for
        for (size_t i1 = k; i1 > 0; i1--) {
            auto i = i1 - 1;
            T K = matrix_clone.get(i, k) / matrix_clone.get(k, k);
            for (size_t j1 = n2; j1 > 0; j1--) {
                auto j = j1 - 1;
                matrix_clone.get(i, j) -= matrix_clone.get(k, j) * K;
            }
        }
    }

    // Отделяем от общей матрицы ответы
    Matrix2D<T> answer{n1, 1};
#pragma omp parallel for
    for (size_t i = 0; i < n1; i++)
        answer.get(i, 0) = matrix_clone.get(i, n2-1);

    return answer;
}