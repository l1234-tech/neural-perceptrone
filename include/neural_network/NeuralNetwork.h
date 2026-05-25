#pragma once
#include "neural_network/Layer.h"
#include "utils/Vector.h"
#include "data/DataSet.h"
#include <vector>
#include <memory>

class NeuralNetwork {
private:
    std::vector<Layer> layers;           
    std::vector<size_t> architecture;    
    
    std::vector<Vector> forwardPassWithCache(const Vector& input);
    double computeLoss(const Vector& output, const Vector& target) const;
    Vector computeOutputGradient(const Vector& output, const Vector& target) const;

public:
    NeuralNetwork(const std::vector<size_t>& architecture);

    Vector forward(const Vector& input) const;

    std::vector<std::vector<double>> backward(const Vector& input, const Vector& target);

    void updateWeights(const std::vector<std::vector<double>>& gradients, double learningRate);

    double trainOneExample(const Vector& input, const Vector& target, double learningRate);

    std::vector<double> getWeights() const;

    void setWeights(const std::vector<double>& newWeights);

    const std::vector<size_t>& getArchitecture() const { return architecture; }

    size_t getNumLayers() const { return layers.size(); }

    const Layer& getLayer(size_t index) const { return layers[index]; }
    Layer& getLayer(size_t index) { return layers[index]; }

    size_t getTotalParameters() const;

    int predict(const Vector& input) const;

    double computeLoss(const DataSet& dataset) const;
    
    // Новая версия с регуляризацией L2
    double computeLoss(const DataSet& dataset, double lambda) const;
    
    // Вычисление штрафа за веса (L2 регуляризация)
    double computeWeightPenalty(double lambda) const;
};