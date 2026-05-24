#include "data/DataSet.h"
#include "neural_network/NeuralNetwork.h"
#include "utils/Vector.h"
#include <algorithm>
#include <random>
#include <stdexcept>

void DataSet::add(const DataPoint& point) {
    points.push_back(point);
}

size_t DataSet::size() const {
    return points.size();
}

const DataPoint& DataSet::get(size_t index) const {
    return points[index];
}
bool DataSet::empty() const {
    return points.empty();
}

double DataSet::computeAccuracy(NeuralNetwork& net) const {
    if (points.empty())
        return 0.0;

    int correct = 0;

    for (const DataPoint& p : points) {
        const auto& features = p.getFeatures();
        Vector input(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            input[i] = features[i];
        }
        
        int predicted = net.predict(input);

        if (predicted == p.getLabel()) {
            correct++;
        }
    }

    return (double)correct / points.size();
}

std::pair<DataSet, DataSet> DataSet::split(double ratio) const {
    if (ratio <= 0.0 || ratio >= 1.0) {
        throw std::invalid_argument("Ratio must be between 0 and 1");
    }
    
    DataSet shuffled = *this;
    shuffled.shuffle();
    
    size_t trainSize = static_cast<size_t>(shuffled.size() * ratio);
    
    DataSet trainSet;
    DataSet testSet;
    
    for (size_t i = 0; i < shuffled.size(); ++i) {
        if (i < trainSize) {
            trainSet.add(shuffled.get(i));
        } else {
            testSet.add(shuffled.get(i));
        }
    }
    
    return {trainSet, testSet};
}

void DataSet::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(points.begin(), points.end(), g);
}

size_t DataSet::getFeatureCount() const {
    if (points.empty()) {
        return 0;
    }
    return points[0].getFeatures().size();
}
