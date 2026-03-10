#include "utils/Matrix.h"
using namespace std;

Matrix::Matrix(size_t n, size_t m): stroks (n), stolbs(m){
    data.resize(stroks); // создаем stroks строк в матрице
    for (size_t i = 0; i < stroks; ++i){
        data[i].resize(stolbs , 0.0);  // для каждой строки создаем stolbs столбцов заполеннных нулями 
    }
}

double& Matrix::at(size_t i, size_t j){
    return data[i][j];
}

double Matrix::at(size_t i, size_t j) const{
    return data[i][j];
}

Matrix Matrix::multipy(const Matrix & a, const Matrix& b){
    Matrix result(a.stroks, b.stolbs);
    for (size_t i = 0 ; i < a.stroks; ++i) {
        for (size_t j =0 ; j < b.stolbs; ++j) {
            double sum = 0.0;
            for (size_t k = 0 ; k < a.stolbs; ++k) {
                sum += a.at(i,k) * b.at(k,j);
            }
            result.at(i,j) = sum;
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix & a, const Matrix& b){
    Matrix result(a.stroks , a.stolbs);
    for (size_t i = 0; i < a.stroks; ++ i) {
        for (size_t j = 0; j < a.stolbs; ++j){
            result.at(i,j) = a.at(i,j) + b.at(i,j);
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result (stolbs, stroks);
    for (size_t i = 0; i < stroks; ++ i) {
        for (size_t j = 0; j < stolbs; ++j){
            result.at(j,i) = data[i][j];
        }
    }
    return result;
}