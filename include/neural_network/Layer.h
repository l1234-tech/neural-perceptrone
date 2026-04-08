#pragma once
#include "utils/Matrix.h"
#include "utils/Vector.h"
#include <vector>

class Layer {
private:
    Matrix weights;    // Веса слоя: [outputSize × inputSize]
    Vector bias;       // Смещения: [outputSize]
    size_t inputSize;  // Сколько входов принимает слой
    size_t outputSize; // Сколько выходов отдаёт слой

public:
    // Создать слой: inputSize входов, outputSize выходов
    // Веса инициализируются случайными числами из диапазона [-1, 1]
    Layer(size_t inputSize, size_t outputSize);

    // Прямой проход: вход -> взвешенная сумма + смещение -> сигмоида -> выход
    Vector forward(const Vector& input) const;

    // Все параметры слоя (веса + смещения) в один плоский вектор
    // Порядок: веса строка за строкой, потом все смещения
    std::vector<double> getWeights() const;

    // Загрузить параметры из плоского вектора (обратная операция к getWeights)
    void setWeights(const std::vector<double>& newWeights);

    // Общее количество настраиваемых параметров (веса + смещения)
    size_t getWeightCount() const;
};
