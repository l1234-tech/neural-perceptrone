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
};