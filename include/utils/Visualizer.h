#pragma once

#include "data/DataSet.h"
#include "neural_network/NeuralNetwork.h"
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

class Visualizer {
public:
    // Сохранить датасет в CSV файл для визуализации
    static void saveDatasetToCSV(const DataSet& dataset, const std::string& filename);
    
    // Сохранить предсказания модели вместе с истинными метками
    static void savePredictionsToCSV(const NeuralNetwork& network, 
                                    const DataSet& dataset, 
                                    const std::string& filename);
    
    // Сохранить историю обучения (loss) в CSV файл
    static void saveTrainingHistoryToCSV(const std::vector<double>& lossHistory,
                                        const std::string& filename);
    
    // Сохранить границу решения для 2D данных
    static void saveDecisionBoundary(const NeuralNetwork& network,
                                    double xMin, double xMax,
                                    double yMin, double yMax,
                                    int gridSize,
                                    const std::string& filename);
    
    // Создать простой отчет о тестировании
    static void saveTestReport(const NeuralNetwork& network,
                              const DataSet& testData,
                              const std::string& filename);
    
private:
    // Вспомогательные методы
    static std::string getCurrentTimestamp();
};
