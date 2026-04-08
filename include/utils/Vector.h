#pragma once
#include <vector>
#include <cmath>

class Vector {
private:
    std::vector<double> data;  // Динамический массив

public:
    Vector(size_t size = 0, double value = 0.0);
    
    double& operator[](size_t i);
    double operator[](size_t i) const;
    size_t size() const;
    
    Vector operator+(const Vector& other) const;
    double dot(const Vector& other) const;  // Скалярное произведение
    
    // Применить функцию (для активации)
    void apply(double (*func)(double));

    Vector(std::initializer_list<double> init);

    // Итераторы для совместимости с STL алгоритмами
    std::vector<double>::iterator begin() { return data.begin(); }
    std::vector<double>::const_iterator begin() const { return data.begin(); }
    std::vector<double>::iterator end() { return data.end(); }
    std::vector<double>::const_iterator end() const { return data.end(); }

    // Метод для преобразования в std::vector<double>
    std::vector<double> toStdVector() const { return data; }
};