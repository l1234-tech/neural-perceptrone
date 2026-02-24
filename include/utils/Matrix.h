#pragma once
#include <vector>
using namespace std;
class Matrix {

private:
    vector<vector<double>> data; // сама матрица, здесь будут записаны веса 
    size_t stroks, stolbs; // это размеры матрицы
public:
    Matrix(size_t n, size_t m); // n - строк, m - столбцов
    double& at(size_t i, size_t j); // чтение и изменение размеров
    double at (size_t i, size_t j) const; // только чтение размеров 

    static Matrix multipy(const Matrix & a, const Matrix& b); // умножение матриц
    static Matrix add(const Matrix & a, const Matrix& b); // сложение матриц 
    Matrix transpose() const; // транспонирование матриц

    size_t get_stolbs() const {return stolbs; } // получение инфы о столбцах
    size_t get_stroks() const {return stroks; } // получение инфы о строках
};