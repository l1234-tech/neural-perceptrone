#include "data/DataLoader.h"
#include <fstream>
#include <sstream>

DataSet DataLoader::loadFromCSV(const std::string& filename) {
    DataSet dataset;
    std::ifstream file(filename);
    std::string line;
    // простой парсер
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double x, y;
        int label;
        char comma;

        if (ss >> x >> comma >> y >> comma >> label) {
            dataset.add({{x, y}, label});
        }
    }
    return dataset;
}
DataSet DataLoader::loadFromCSVNoLabels(const std::string& filename) {
    DataSet dataset;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double x, y;
        char comma;

        if (ss >> x >> comma >> y) {
            dataset.add({{x, y}, -1}); // здесь label = -1 , потому что мы еще не знаем класс точки  
        }
    }
    return dataset;
}
// void DataLoader::saveToCSV(const DataSet& data, const std::string& filename){

// }