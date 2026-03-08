#include "DataPoint.h"

DataPoint::DataPoint(const std::vector<double>& feats, int lbl)
    : features(feats), label(lbl) {}

const std::vector<double>& DataPoint::getFeatures() const {
    return features;
}

int DataPoint::getLabel() const {
    return label;
}