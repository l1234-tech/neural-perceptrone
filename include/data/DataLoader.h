#pragma once
#include "data/DataSet.h"
#include <string>
#include <string>
#include <fstream>
#include <memory>

// Универсальный парсер CSV с произвольным количеством признаков
class UniversalCSVParser {
public:
    virtual ~UniversalCSVParser() = default;
    virtual DataPoint parse(const std::string& line, char delimiter) const = 0;
    virtual bool validateLine(const std::string& line, char delimiter) const = 0;
};

// Основной класс загрузчика данных
class DataLoader {
private:
    std::string filename;
    std::unique_ptr<UniversalCSVParser> parser;
    char delimiter;
    bool hasHeader;
    
    std::ifstream openFileForReading() const;
    std::ofstream openFileForWriting() const;
    
public:
    DataLoader(const std::string& filename, bool hasLabels = true, char delimiter = ',', bool hasHeader = true);
    ~DataLoader() = default;
    
    // Запрещаем копирование
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;
    
    // Разрешаем перемещение
    DataLoader(DataLoader&&) = default;
    DataLoader& operator=(DataLoader&&) = default;
    
    DataSet load();
    void save(const DataSet& data);
    void saveNoLabels(const DataSet& data);
    
    void setDelimiter(char newDelimiter) { delimiter = newDelimiter; }
    char getDelimiter() const { return delimiter; }
    
    // Статические методы для обратной совместимости
    static DataSet loadFromCSV(const std::string& filename);
    static DataSet loadFromCSVNoLabels(const std::string& filename);
    static void saveToCSV(const DataSet& data, const std::string& filename);
    static void saveToCSVNoLabels(const DataSet& data, const std::string& filename);
    
    // Новые методы
    static size_t detectFeatureCount(const std::string& filename, char delimiter = ',', bool hasHeader = true);
    static bool hasHeaderRow(const std::string& filename, char delimiter = ',');
};