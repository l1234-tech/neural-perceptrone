#include "data/DataLoader.h"
#include "data/DataPoint.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>

// Реализации конкретных парсеров
class CSVWithLabelsParser : public DataParser {
public:
    DataPoint parse(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        int label;
        char comma;
        
        if (!(ss >> x >> comma >> y >> comma >> label)) {
            throw std::runtime_error("Failed to parse line with labels: " + line);
        }
        
        if (comma != delimiter) {
            throw std::runtime_error("Invalid delimiter in line: " + line);
        }
        
        return DataPoint({x, y}, label);
    }
    
    bool validateLine(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        int label;
        char comma;
        
        return !!(ss >> x >> comma >> y >> comma >> label) && comma == delimiter;
    }
};

class CSVWithoutLabelsParser : public DataParser {
public:
    DataPoint parse(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        char comma;
        
        if (!(ss >> x >> comma >> y)) {
            throw std::runtime_error("Failed to parse line without labels: " + line);
        }
        
        if (comma != delimiter) {
            throw std::runtime_error("Invalid delimiter in line: " + line);
        }
        
        return DataPoint({x, y}, -1);
    }
    
    bool validateLine(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        char comma;
        
        return !!(ss >> x >> comma >> y) && comma == delimiter;
    }
};

std::ifstream DataLoader::openFileForReading() const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    return file;
}

std::ofstream DataLoader::openFileForWriting() const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    return file;
}

DataLoader::DataLoader(const std::string& filename, bool hasLabels, char delimiter)
    : filename(filename), delimiter(delimiter) {
    
    if (hasLabels) {
        parser = std::make_unique<CSVWithLabelsParser>();
    } else {
        parser = std::make_unique<CSVWithoutLabelsParser>();
    }
}

DataSet DataLoader::load() {
    DataSet dataset;
    std::ifstream file = openFileForReading();
    std::string line;
    
    size_t lineNumber = 0;
    size_t validLines = 0;
    size_t invalidLines = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        if (line.empty()) {
            continue;
        }
        
        try {
            DataPoint point = parser->parse(line, delimiter);
            dataset.add(point);
            validLines++;
        } catch (const std::exception& e) {
            invalidLines++;
        }
    }
    
    if (validLines == 0) {
        throw std::runtime_error("No valid data loaded from file: " + filename);
    }
    
    return dataset;
}

void DataLoader::save(const DataSet& data) {
    if (data.empty()) {
        throw std::runtime_error("Cannot save empty dataset to file: " + filename);
    }
    
    std::ofstream file = openFileForWriting();
    
    for (size_t i = 0; i < data.size(); ++i) {
        const DataPoint& point = data.get(i);
        const std::vector<double>& features = point.getFeatures();
        
        if (features.size() < 2) {
            throw std::runtime_error("DataPoint must have at least 2 features for CSV format");
        }
        
        file << features[0] << delimiter 
             << features[1] << delimiter
             << point.getLabel();
        
        if (i < data.size() - 1) {
            file << "\n";
        }
    }
}

DataSet DataLoader::loadFromCSV(const std::string& filename) {
    DataLoader loader(filename, true, ',');
    return loader.load();
}

DataSet DataLoader::loadFromCSVNoLabels(const std::string& filename) {
    DataLoader loader(filename, false, ',');
    return loader.load();
}

void DataLoader::saveToCSV(const DataSet& data, const std::string& filename) {
    DataLoader loader(filename, true, ',');
    loader.save(data);
}