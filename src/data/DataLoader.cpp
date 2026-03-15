#include "data/DataLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>

// ============================================================================
// Классы парсеров
// ============================================================================

// Базовый класс парсера
class DataParser {
public:
    virtual ~DataParser() = default;
    virtual DataPoint parse(const std::string& line, char delimiter) const = 0;
    virtual bool validateLine(const std::string& line, char delimiter) const = 0;
};

// Парсер для CSV с метками классов
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
        
        // Проверяем, что разделитель правильный
        if (comma != delimiter) {
            throw std::runtime_error("Invalid delimiter in line: " + line);
        }
        
        return {{x, y}, label};
    }
    
    bool validateLine(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        int label;
        char comma;
        
        return !!(ss >> x >> comma >> y >> comma >> label) && comma == delimiter;
    }
};

// Парсер для CSV без меток классов
class CSVWithoutLabelsParser : public DataParser {
public:
    DataPoint parse(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        char comma;
        
        if (!(ss >> x >> comma >> y)) {
            throw std::runtime_error("Failed to parse line without labels: " + line);
        }
        
        // Проверяем, что разделитель правильный
        if (comma != delimiter) {
            throw std::runtime_error("Invalid delimiter in line: " + line);
        }
        
        return {{x, y}, -1}; // label = -1 для данных без меток
    }
    
    bool validateLine(const std::string& line, char delimiter) const override {
        std::stringstream ss(line);
        double x, y;
        char comma;
        
        return !!(ss >> x >> comma >> y) && comma == delimiter;
    }
};

// ============================================================================
// Вспомогательные методы DataLoader
// ============================================================================

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

// ============================================================================
// Конструктор DataLoader
// ============================================================================

DataLoader::DataLoader(const std::string& filename, bool hasLabels, char delimiter)
    : filename(filename), delimiter(delimiter) {
    
    // Инициализируем соответствующий парсер
    if (hasLabels) {
        parser = std::make_unique<CSVWithLabelsParser>();
    } else {
        parser = std::make_unique<CSVWithoutLabelsParser>();
    }
}

// ============================================================================
// Основные методы DataLoader
// ============================================================================

DataSet DataLoader::load() {
    DataSet dataset;
    std::ifstream file = openFileForReading();
    std::string line;
    
    size_t lineNumber = 0;
    size_t validLines = 0;
    size_t invalidLines = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Пропускаем пустые строки
        if (line.empty()) {
            continue;
        }
        
        try {
            // Парсим строку с помощью соответствующего парсера
            DataPoint point = parser->parse(line, delimiter);
            dataset.add(point);
            validLines++;
        } catch (const std::exception& e) {
            invalidLines++;
            // Можно добавить логирование ошибок, но не прерываем загрузку
            // std::cerr << "Warning: Line " << lineNumber << ": " << e.what() << std::endl;
        }
    }
    
    // Проверяем, что загружены какие-то данные
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
        
        // Записываем данные в формате: x, y, label
        file << point.features[0] << delimiter 
             << point.features[1] << delimiter
             << point.label;
        
        // Добавляем перевод строки, если это не последняя точка
        if (i < data.size() - 1) {
            file << "\n";
        }
    }
}

// ============================================================================
// Статические методы для обратной совместимости
// ============================================================================

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