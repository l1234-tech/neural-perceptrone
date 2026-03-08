#pragma once
#include <vector>

class DataPoint {
private:
    std::vector<double> features;
    int label;

public:
    DataPoint(const std::vector<double>& feats, int lbl);

    const std::vector<double>& getFeatures() const;

    int getLabel() const;
};