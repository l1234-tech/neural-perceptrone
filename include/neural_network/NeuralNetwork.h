#pragma once
#include "neural_network/Layer.h"
#include "utils/Vector.h"
#include "data/DataSet.h"
#include <vector>
#include <memory>

// Класс нейронной сети
// Состоит из последовательности слоёв, поддерживает прямой проход и обучение
class NeuralNetwork {
private:
    std::vector<Layer> layers;           // Все слои сети
    std::vector<size_t> architecture;    // Архитектура: [input, hidden1, ..., output]
    
    // Вспомогательные методы для обучения
    std::vector<Vector> forwardPassWithCache(const Vector& input);
    double computeLoss(const Vector& output, const Vector& target) const;
    Vector computeOutputGradient(const Vector& output, const Vector& target) const;

public:
    // Конструктор: задаёт архитектуру сети
    // Пример: NeuralNetwork net({2, 4, 1}); // 2 входа, 4 нейрона в скрытом слое, 1 выход
    NeuralNetwork(const std::vector<size_t>& architecture);

    // Прямой проход: вход -> все слои -> выход
    Vector forward(const Vector& input) const;

    // Обратное распространение ошибки (backpropagation)
    // Возвращает градиенты для всех параметров сети
    std::vector<std::vector<double>> backward(const Vector& input, const Vector& target);

    // Обновить веса с помощью градиентов и скорости обучения
    void updateWeights(const std::vector<std::vector<double>>& gradients, double learningRate);

    // Обучение на одном примере (один шаг градиентного спуска)
    double trainOneExample(const Vector& input, const Vector& target, double learningRate);

    // Получить все параметры сети в одном плоском векторе
    std::vector<double> getWeights() const;

    // Установить все параметры сети из плоского вектора
    void setWeights(const std::vector<double>& newWeights);

    // Получить архитектуру сети
    const std::vector<size_t>& getArchitecture() const { return architecture; }

    // Получить количество слоёв
    size_t getNumLayers() const { return layers.size(); }

    // Получить слой по индексу
    const Layer& getLayer(size_t index) const { return layers[index]; }
    Layer& getLayer(size_t index) { return layers[index]; }

    // Получить общее количество параметров
    size_t getTotalParameters() const;

    // Предсказать класс для входного вектора
    int predict(const Vector& input) const;

    // Вычислить потери (loss) на датасете
    double computeLoss(const DataSet& dataset) const;
};