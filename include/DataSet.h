#pragma once

#include "DataPoint.h"
#include <vector>
#include <cstddef>

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
};