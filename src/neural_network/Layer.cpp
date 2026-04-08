#include "neural_network/Layer.h"
#include "neural_network/Activation.h"
#include "utils/Random.h"

Layer::Layer(size_t inputSize, size_t outputSize)
    : weights(outputSize, inputSize),
      bias(outputSize, 0.0),
      inputSize(inputSize),
      outputSize(outputSize)
{
    // диапазон зависит от количества входов,
    // чтобы сигнал не затухал и не взрывался при прохождении через слои
    double range = 1.0 / std::sqrt(static_cast<double>(inputSize));

    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weights.at(i, j) = randomDouble(-range, range);
        }
        bias[i] = randomDouble(-range, range);
    }
}

Vector Layer::forward(const Vector& input) const{
    Vector output(outputSize);

    for (size_t i = 0; i < outputSize; ++i) {
        double sum = bias[i];
        for (size_t j = 0; j < inputSize; ++j) {
            sum += weights.at(i, j) * input[j];
        }
        output[i] = Activation::sigmoid(sum);
    }

    return output;
}

std::vector<double> Layer::getWeights() const {
    std::vector<double> result;
    result.reserve(getWeightCount());

    // Сначала все веса, строка за строкой
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            result.push_back(weights.at(i, j));
        }
    }

    // Затем все смещения
    for (size_t i = 0; i < outputSize; ++i) {
        result.push_back(bias[i]);
    }

    return result;
}

void Layer::setWeights(const std::vector<double>& newWeights) {
    size_t idx = 0;

    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weights.at(i, j) = newWeights[idx++];
        }
    }

    for (size_t i = 0; i < outputSize; ++i) {
        bias[i] = newWeights[idx++];
    }
}

size_t Layer::getWeightCount() const {
    return outputSize * inputSize + outputSize;
}
