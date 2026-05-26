#include "data/DataLoader.h"
#include "data/DataPoint.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <iostream>

// Universal CSV parser with support for arbitrary number of features
class CSVWithLabelsParser : public UniversalCSVParser {
public:
    DataPoint parse(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> features;
        
        // Read all comma-separated values
        while (std::getline(ss, token, delimiter)) {
            // Trim spaces from beginning and end
            size_t start = token.find_first_not_of(" 	\r\n");
            size_t end = token.find_last_not_of(" 	\r\n");
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
            }
            
            try {
                double value = std::stod(token);
                features.push_back(value);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to parse value: '" + token + "' in line: " + line);
            }
        }
        
        if (features.size() < 2) {
            throw std::runtime_error("Line must have at least 2 columns (features + target): " + line);
        }
        
        // Last column is target, others are features
        double targetValue = features.back();
        int label = static_cast<int>(std::round(targetValue));
        
        // Check if label is close to integer
        if (std::abs(targetValue - label) > 1e-6) {
            std::cerr << "Warning: Label " << targetValue << " is not integer, rounding to " << label << std::endl;
        }
        
        features.pop_back();
        
        return DataPoint(features, label);
    }
    
    bool validateLine(const std::string& line, char delimiter) const override {
        if (line.empty()) return false;
        
        std::stringstream ss(line);
        std::string token;
        std::vector<double> values;
        
        while (std::getline(ss, token, delimiter)) {
            size_t start = token.find_first_not_of(" 	\r\n");
            size_t end = token.find_last_not_of(" 	\r\n");
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
            }
            
            try {
                values.push_back(std::stod(token));
            } catch (...) {
                return false;
            }
        }
        
        return values.size() >= 2;
    }
};

class CSVWithoutLabelsParser : public UniversalCSVParser {
public:
    DataPoint parse(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> features;
        
        while (std::getline(ss, token, delimiter)) {
            size_t start = token.find_first_not_of(" 	\r\n");
            size_t end = token.find_last_not_of(" 	\r\n");
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
            }
            
            try {
                features.push_back(std::stod(token));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to parse value: '" + token + "' in line: " + line);
            }
        }
        
        return DataPoint(features, -1);
    }
    
    bool validateLine(const std::string& line, char delimiter) const override {
        if (line.empty()) return false;
        
        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, delimiter)) {
            size_t start = token.find_first_not_of(" 	\r\n");
            size_t end = token.find_last_not_of(" 	\r\n");
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
            }
            
            try {
                std::stod(token);
            } catch (...) {
                return false;
            }
        }
        
        return true;
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

DataLoader::DataLoader(const std::string& filename, bool hasLabels, char delimiter, bool hasHeader)
    : filename(filename), delimiter(delimiter), hasHeader(hasHeader) {
    
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
    size_t expectedFeatureCount = 0;
    bool firstValidLine = true;
    
    // Skip header if present
    if (hasHeader) {
        std::getline(file, line);
        lineNumber++;
    }
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        if (line.empty()) {
            continue;
        }
        
        try {
            DataPoint point = parser->parse(line, delimiter);
            
            // Check feature count consistency
            size_t currentFeatureCount = point.getFeatures().size();
            if (firstValidLine) {
                expectedFeatureCount = currentFeatureCount;
                firstValidLine = false;
            } else if (currentFeatureCount != expectedFeatureCount) {
                std::cerr << "Warning: Line " << lineNumber << " has " << currentFeatureCount 
                          << " features, expected " << expectedFeatureCount << std::endl;
                // Continue but add the point
            }
            
            dataset.add(point);
            validLines++;
        } catch (const std::exception& e) {
            invalidLines++;
            std::cerr << "Error parsing line " << lineNumber << ": " << e.what() << std::endl;
        }
    }
    
    if (validLines == 0) {
        throw std::runtime_error("No valid data loaded from file: " + filename);
    }
    
    if (invalidLines > 0) {
        std::cerr << "Warning: " << invalidLines << " invalid lines skipped in file: " << filename << std::endl;
    }
    
    std::cout << "Loaded " << validLines << " valid samples with " << expectedFeatureCount << " features" << std::endl;
    
    return dataset;
}

void DataLoader::save(const DataSet& data) {
    if (data.empty()) {
        throw std::runtime_error("Cannot save empty dataset to file: " + filename);
    }
    
    std::ofstream file = openFileForWriting();
    
    // Write header
    size_t featureCount = data.getFeatureCount();
    for (size_t i = 0; i < featureCount; ++i) {
        file << "feature_" << i;
        if (i < featureCount - 1) {
            file << delimiter;
        }
    }
    file << delimiter << "target\n";
    
    // Write data
    for (size_t i = 0; i < data.size(); ++i) {
        const DataPoint& point = data.get(i);
        const std::vector<double>& features = point.getFeatures();
        
        for (size_t j = 0; j < features.size(); ++j) {
            file << features[j];
            if (j < features.size() - 1) {
                file << delimiter;
            }
        }
        
        file << delimiter << point.getLabel();
        
        if (i < data.size() - 1) {
            file << "\n";
        }
    }
}

void DataLoader::saveNoLabels(const DataSet& data) {
    if (data.empty()) {
        throw std::runtime_error("Cannot save empty dataset to file: " + filename);
    }
    
    std::ofstream file = openFileForWriting();
    
    // Write header
    size_t featureCount = data.getFeatureCount();
    for (size_t i = 0; i < featureCount; ++i) {
        file << "feature_" << i;
        if (i < featureCount - 1) {
            file << delimiter;
        }
    }
    file << "\n";
    
    // Write data (only features)
    for (size_t i = 0; i < data.size(); ++i) {
        const DataPoint& point = data.get(i);
        const std::vector<double>& features = point.getFeatures();
        
        for (size_t j = 0; j < features.size(); ++j) {
            file << features[j];
            if (j < features.size() - 1) {
                file << delimiter;
            }
        }
        
        if (i < data.size() - 1) {
            file << "\n";
        }
    }
}

size_t DataLoader::detectFeatureCount(const std::string& filename, char delimiter, bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    
    // Skip header if present
    if (hasHeader) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("File is empty: " + filename);
        }
    }
    
    // Read first data line
    if (!std::getline(file, line)) {
        throw std::runtime_error("No data lines in file: " + filename);
    }
    
    // Count columns
    std::stringstream ss(line);
    std::string token;
    size_t columnCount = 0;
    
    while (std::getline(ss, token, delimiter)) {
        columnCount++;
    }
    
    // Last column is target, others are features
    if (columnCount < 2) {
        throw std::runtime_error("File must have at least 2 columns (features + target)");
    }
    
    return columnCount - 1; // Minus target
}

bool DataLoader::hasHeaderRow(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    if (!std::getline(file, line)) {
        return false;
    }
    
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        size_t start = token.find_first_not_of(" 	\r\n");
        size_t end = token.find_last_not_of(" 	\r\n");
        if (start != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }
        
        // If at least one token contains non-numeric characters (except dot and minus), it's a header
        bool isNumeric = true;
        bool hasDigit = false;
        
        for (char c : token) {
            if (std::isdigit(c)) {
                hasDigit = true;
            } else if (c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E') {
                isNumeric = false;
                break;
            }
        }
        
        if (!isNumeric || !hasDigit) {
            return true;
        }
    }
    
    return false;
}

DataSet DataLoader::loadFromCSV(const std::string& filename) {
    bool header = hasHeaderRow(filename);
    DataLoader loader(filename, true, ',', header);
    return loader.load();
}

DataSet DataLoader::loadFromCSVNoLabels(const std::string& filename) {
    bool header = hasHeaderRow(filename);
    DataLoader loader(filename, false, ',', header);
    return loader.load();
}

void DataLoader::saveToCSV(const DataSet& data, const std::string& filename) {
    DataLoader loader(filename, true, ',', true);
    loader.save(data);
}

void DataLoader::saveToCSVNoLabels(const DataSet& data, const std::string& filename) {
    DataLoader loader(filename, false, ',', true);
    loader.saveNoLabels(data);
}