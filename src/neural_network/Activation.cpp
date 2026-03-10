#include "neural_network/Activation.h"
#include <cmath>
#include <stdexcept>

double Activation::sigmoid(double x) {
    // Защита от переполнения exp():
    // при x < -500 результат практически 0,
    // при x > 500 — практически 1
    if (x < -500.0) return 0.0;
    if (x > 500.0) return 1.0;

    return 1.0 / (1.0 + std::exp(-x));
}

double Activation::relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

int Activation::step(double x, double threshold) {
    return (x >= threshold) ? 1 : 0;
}

int Activation::argmax(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("argmax: вектор пуст");
    }

    int maxIndex = 0;
    double maxValue = values[0];

    for (int i = 1; i < static_cast<int>(values.size()); ++i) {
        if (values[i] > maxValue) {
            maxValue = values[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}
