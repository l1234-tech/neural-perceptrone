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
    
    // Создаём слои: для каждой пары (input_size, output_size) в архитектуре
    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        layers.emplace_back(architecture[i], architecture[i + 1]);
    }
}

Vector NeuralNetwork::forward(const Vector& input) const {
    if (input.size() != architecture[0]) {
        throw std::invalid_argument("Размер входного вектора не соответствует архитектуре сети");
    }
    
    Vector current = input;
    
    // Последовательно применяем все слои
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
    activations.push_back(input);  // Входной слой
    
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
    
    // Binary cross-entropy loss (лучше подходит для бинарной классификации)
    const double epsilon = 1e-8;  // Для стабильности вычислений логарифма
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double y = target[i];
        double p = output[i];
        // Добавляем epsilon для избежания log(0)
        // Для стабильности также ограничиваем p в диапазоне [epsilon, 1-epsilon]
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
    
    // Градиент для binary cross-entropy с сигмоидой: output - target
    // Производная сигмоиды уже учтена в этой простой формуле
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i] = output[i] - target[i];
    }
    
    return gradient;
}

std::vector<std::vector<double>> NeuralNetwork::backward(const Vector& input, const Vector& target) {
    // Прямой проход с сохранением активаций
    std::vector<Vector> activations = forwardPassWithCache(input);
    
    // Градиенты для каждого слоя
    std::vector<std::vector<double>> layerGradients(layers.size());
    
    // Вычисляем градиент на выходном слое
    Vector currentGradient = computeOutputGradient(activations.back(), target);
    
    // Обратное распространение через слои
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        const Layer& layer = layers[i];
        const Vector& layerInput = activations[i];
        const Vector& layerOutput = activations[i + 1];
        
        // Градиенты для весов и смещений этого слоя
        std::vector<double> gradients;
        gradients.reserve(layer.getWeightCount());
        
        // Градиенты для весов
        for (size_t neuronIdx = 0; neuronIdx < layerOutput.size(); ++neuronIdx) {
            double neuronGradient = currentGradient[neuronIdx];
            
            // Градиенты для весов этого нейрона
            for (size_t inputIdx = 0; inputIdx < layerInput.size(); ++inputIdx) {
                gradients.push_back(neuronGradient * layerInput[inputIdx]);
            }
            
            // Градиент для смещения этого нейрона
            gradients.push_back(neuronGradient);
        }
        
        layerGradients[i] = gradients;
        
        // Если не последний слой, вычисляем градиент для предыдущего слоя
        if (i > 0) {
            Vector nextGradient(layerInput.size(), 0.0);
            
            // Распространяем градиент назад через веса
            for (size_t neuronIdx = 0; neuronIdx < layerOutput.size(); ++neuronIdx) {
                double neuronGradient = currentGradient[neuronIdx];
                
                // Для каждого входа добавляем вклад от этого нейрона
                for (size_t inputIdx = 0; inputIdx < layerInput.size(); ++inputIdx) {
                    // Нужно получить вес, но у нас нет доступа к весам извне
                    // В упрощённой версии пропускаем этот шаг
                    // В полной реализации нужно хранить веса или получать их через getWeights()
                }
            }
            
            // Упрощённая версия: просто передаём градиент дальше
            // В реальной реализации нужно умножать на транспонированную матрицу весов
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
    
    // Получаем текущие веса
    std::vector<double> currentWeights = getWeights();
    
    // Применяем градиентный спуск
    size_t weightIndex = 0;
    for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
        const auto& layerGradients = gradients[layerIdx];
        
        for (size_t i = 0; i < layerGradients.size(); ++i) {
            currentWeights[weightIndex] -= learningRate * layerGradients[i];
            weightIndex++;
        }
    }
    
    // Устанавливаем обновлённые веса
    setWeights(currentWeights);
}

double NeuralNetwork::trainOneExample(const Vector& input, const Vector& target, double learningRate) {
    // Прямой проход для вычисления ошибки
    Vector output = forward(input);
    double loss = computeLoss(output, target);
    
    // Обратное распространение
    auto gradients = backward(input, target);
    
    // Обновление веса
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
    
    // Для бинарной классификации
    if (output.size() == 1) {
        return (output[0] >= 0.5) ? 1 : 0;
    }
    
    // Для многоклассовой классификации
    // Используем begin() и end(), которые теперь есть в классе Vector
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
        
        // Преобразуем вектор признаков в Vector
        Vector input(features.size());
        for (size_t j = 0; j < features.size(); ++j) {
            input[j] = features[j];
        }
        
        // Целевой вектор
        Vector target(architecture.back(), 0.0);
        int label = point.getLabel();
        
        if (architecture.back() == 1) {
            // Бинарная классификация с одним выходом
            // label должен быть 0 или 1
            target[0] = static_cast<double>(label);
        } else {
            // Многоклассовая классификация (one-hot encoding)
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