#include "neural_network/NeuralNetwork.h"
#include "neural_network/Activation.h"
#include "neural_network/Layer.h"
#include "data/DataSet.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& architecture)
    : architecture(architecture) {
    
    if (architecture.size() < 2) {
        throw std::invalid_argument("Архитектура должна содержать минимум входной и выходной слои");
    }
    
    // Создаём слои: для каждой пары (input_size, output_size)
    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        layers.emplace_back(architecture[i], architecture[i + 1]);
    }
}

Vector NeuralNetwork::forward(const Vector& input) const {
    if (input.size() != architecture[0]) {
        throw std::invalid_argument("Размер входного вектора не соответствует архитектуре сети");
    }
    
    Vector current = input;
    
    for (const auto& layer : layers) {
        current = layer.forward(current);
    }
    
    return current;
}

std::vector<Vector> NeuralNetwork::forwardPassWithCache(const Vector& input) {
    if (input.size() != architecture[0]) {
        throw std::invalid_argument("Размер входного вектора не соответствует архитектуре сети");
    }
    
    std::vector<Vector> activations;
    activations.reserve(layers.size() + 1);
    activations.push_back(input); 
    
    Vector current = input;
    
    // Сохраняем активации каждого слоя для обратного распространения
    for (const auto& layer : layers) {
        current = layer.forward(current);
        activations.push_back(current);
    }
    
    return activations;
}

double NeuralNetwork::computeLoss(const Vector& output, const Vector& target) const {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Размеры выходного вектора и целевого вектора не совпадают");
    }
    
    const double epsilon = 1e-8; 
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double y = target[i];
        double p = output[i];
        // ограничиваем p в диапазоне [epsilon, 1-epsilon]
        p = std::max(epsilon, std::min(1.0 - epsilon, p));
        loss += - (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }
    
    return loss / output.size();
}

Vector NeuralNetwork::computeOutputGradient(const Vector& output, const Vector& target) const {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Размеры выходного вектора и целевого вектора не совпадают");
    }
    
    Vector gradient(output.size());
    
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i] = output[i] - target[i];
    }
    
    return gradient;
}

std::vector<std::vector<double>> NeuralNetwork::backward(const Vector& input, const Vector& target) {

    std::vector<Vector> activations = forwardPassWithCache(input);

    std::vector<std::vector<double>> layerGradients(layers.size());

    Vector currentGradient = computeOutputGradient(activations.back(), target);

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        const Layer& layer = layers[i];
        const Vector& layerInput = activations[i];
        const Vector& layerOutput = activations[i + 1];

        std::vector<double> gradients;
        gradients.reserve(layer.getWeightCount());

        for (size_t neuronIdx = 0; neuronIdx < layerOutput.size(); ++neuronIdx) {
            double neuronGradient = currentGradient[neuronIdx];
   
            for (size_t inputIdx = 0; inputIdx < layerInput.size(); ++inputIdx) {
                gradients.push_back(neuronGradient * layerInput[inputIdx]);
            }

            gradients.push_back(neuronGradient);
        }
        
        layerGradients[i] = gradients;

        if (i > 0) {
            Vector nextGradient(layerInput.size(), 0.0);

            for (size_t neuronIdx = 0; neuronIdx < layerOutput.size(); ++neuronIdx) {
                double neuronGradient = currentGradient[neuronIdx];
                
                for (size_t inputIdx = 0; inputIdx < layerInput.size(); ++inputIdx) {
                }
            }
            
            currentGradient = Vector(layerInput.size(), 0.0);
            for (size_t j = 0; j < std::min(currentGradient.size(), nextGradient.size()); ++j) {
                currentGradient[j] = nextGradient[j];
            }
        }
    }
    
    return layerGradients;
}

void NeuralNetwork::updateWeights(const std::vector<std::vector<double>>& gradients, double learningRate) {
    if (gradients.size() != layers.size()) {
        throw std::invalid_argument("Количество градиентов не соответствует количеству слоёв");
    }
    
    std::vector<double> currentWeights = getWeights();

    size_t weightIndex = 0;
    for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
        const auto& layerGradients = gradients[layerIdx];
        
        for (size_t i = 0; i < layerGradients.size(); ++i) {
            currentWeights[weightIndex] -= learningRate * layerGradients[i];
            weightIndex++;
        }
    }
    
    setWeights(currentWeights);
}

double NeuralNetwork::trainOneExample(const Vector& input, const Vector& target, double learningRate) {
    Vector output = forward(input);
    double loss = computeLoss(output, target);
    
    auto gradients = backward(input, target);
    
    updateWeights(gradients, learningRate);
    
    return loss;
}

std::vector<double> NeuralNetwork::getWeights() const {
    std::vector<double> allWeights;
    
    for (const auto& layer : layers) {
        auto layerWeights = layer.getWeights();
        allWeights.insert(allWeights.end(), layerWeights.begin(), layerWeights.end());
    }
    
    return allWeights;
}

void NeuralNetwork::setWeights(const std::vector<double>& newWeights) {
    std::vector<double> remainingWeights = newWeights;
    
    for (auto& layer : layers) {
        size_t layerWeightCount = layer.getWeightCount();
        
        if (remainingWeights.size() < layerWeightCount) {
            throw std::invalid_argument("Недостаточно весов для установки");
        }
        
        std::vector<double> layerWeights(remainingWeights.begin(), 
                                        remainingWeights.begin() + layerWeightCount);
        layer.setWeights(layerWeights);
        
        // Удаляем использованные веса
        remainingWeights.erase(remainingWeights.begin(), 
                             remainingWeights.begin() + layerWeightCount);
    }
    
    if (!remainingWeights.empty()) {
        throw std::invalid_argument("Остались лишние веса после установки");
    }
}

size_t NeuralNetwork::getTotalParameters() const {
    size_t total = 0;
    
    for (const auto& layer : layers) {
        total += layer.getWeightCount();
    }
    
    return total;
}

int NeuralNetwork::predict(const Vector& input) const {
    Vector output = forward(input);
    
    if (output.size() == 1) {
        return (output[0] >= 0.5) ? 1 : 0;
    }
    
    return Activation::argmax(std::vector<double>(output.begin(), output.end()));
}

double NeuralNetwork::computeLoss(const DataSet& dataset) const {
    if (dataset.empty()) {
        return 0.0;
    }
    
    double totalLoss = 0.0;
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& point = dataset.get(i);
        const auto& features = point.getFeatures();
        
        Vector input(features.size());
        for (size_t j = 0; j < features.size(); ++j) {
            input[j] = features[j];
        }
        
        Vector target(architecture.back(), 0.0);
        int label = point.getLabel();
        
        if (architecture.back() == 1) {
            // Бинарная классификация с одним выходом
            // label должен быть 0 или 1
            target[0] = static_cast<double>(label);
        } else {
            // Многоклассовая классификация 
            if (label >= 0 && static_cast<size_t>(label) < target.size()) {
                target[label] = 1.0;
            }
        }
        
        // Прямой проход и вычисление ошибки
        Vector output = forward(input);
        totalLoss += computeLoss(output, target);
    }
    
    return totalLoss / dataset.size();
}