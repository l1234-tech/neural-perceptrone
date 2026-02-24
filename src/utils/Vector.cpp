#include "Vector.h"
#include "Random.h"

// Конструктор: создаёт вектор из 'size' элементов, заполненных 'value'
Vector::Vector(size_t size, double value) {
    data.resize(size, value);
}

// Доступ для записи
double& Vector::operator[](size_t i) {
    return data[i];
}

// Доступ для чтения
double Vector::operator[](size_t i) const {
    return data[i];
}

// Размер вектора
size_t Vector::size() const {
    return data.size();
}

// Сложение векторов
Vector Vector::operator+(const Vector& other) const {
    Vector result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] + other[i];
    }
    return result;
}

// Скалярное произведение
double Vector::dot(const Vector& other) const {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i] * other[i];
    }
    return sum;
}

// Применить функцию к каждому элементу
void Vector::apply(double (*func)(double)) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = func(data[i]);
    }
}