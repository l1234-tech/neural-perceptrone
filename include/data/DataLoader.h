#pragma once
#include "data/DataSet.h"
#include <string>

class DataLoader {
public:
    static DataSet loadFromCSV(const std::string& filename);

    static DataSet loadFromCSVNoLabels(const std::string& filename);
    // для предсказания (точки без меток класса)
    static void saveToCSV(const DataSet& data, const std::string& filename);
    // в основном для записи данных из консоли (потом для графиков и для результатов)
};