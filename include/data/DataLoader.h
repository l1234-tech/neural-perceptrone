#pragma once
#include "data/DataSet.h"
#include <string>
#include <memory>
#include <fstream>

class DataParser;

class DataLoader {
private:
    std::string filename;
    std::unique_ptr<DataParser> parser;
    char delimiter;
    
    std::ifstream openFileForReading() const;
    std::ofstream openFileForWriting() const;
    
public:
    DataLoader(const std::string& filename, bool hasLabels = true, char delimiter = ',');
    ~DataLoader() = default;
    
    // Запрещаем копирование
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;
    
    // Разрешаем перемещение
    DataLoader(DataLoader&&) = default;
    DataLoader& operator=(DataLoader&&) = default;
    
    DataSet load();
    void save(const DataSet& data);
    
    void setDelimiter(char newDelimiter) { delimiter = newDelimiter; }
    char getDelimiter() const { return delimiter; }
    
    // Статические методы для обратной совместимости
    static DataSet loadFromCSV(const std::string& filename);
    static DataSet loadFromCSVNoLabels(const std::string& filename);
    static void saveToCSV(const DataSet& data, const std::string& filename);
};