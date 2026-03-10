#include "data/DataSet.h"
//#include "NeuralNetwork.h"

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

// double DataSet::computeAccuracy(NeuralNetwork& net) const {

//     if (points.empty())
//         return 0.0;

//     int correct = 0;

//     for (const DataPoint& p : points) {

//         int predicted = net.predict(p.getFeatures());

//         if (predicted == p.getLabel()) {
//             correct++;
//         }
//     }

//     return (double)correct / points.size();
// }