#pragma once

#include "data/DataPoint.h"
#include <vector>
#include <cstddef>
#include <utility>

class NeuralNetwork;

class DataSet {
private:
    std::vector<DataPoint> points;

public:
    DataSet() = default;

    void add(const DataPoint& point);
    size_t size() const;
    const DataPoint& get(size_t index) const;
    bool empty() const;
    double computeAccuracy(NeuralNetwork& net) const;
    
    std::pair<DataSet, DataSet> split(double ratio = 0.8) const;
    void shuffle();
    size_t getFeatureCount() const;
};