#pragma once

#include "data/DataSet.h"
#include "neural_network/NeuralNetwork.h"
#include <vector>
#include <string>
#include <fstream>

class Visualizer {
public:
    // Визуализация 2D датасета с предсказаниями сети
    // Сохраняет данные для построения в CSV файлы и генерирует Python скрипт для визуализации
    static void plot2D(const DataSet& data, const NeuralNetwork& net, 
                      const std::string& outputPath);
    
    // Визуализация многомерных данных (снижение размерности)
    // Заглушка для будущей реализации
    static void plotMultiDimensional(const DataSet& data, const NeuralNetwork& net,
                                   const std::string& outputPath);
    
    // График истории обучения
    // Сохраняет историю обучения в CSV файл и генерирует Python скрипт для визуализации
    static void plotTrainingHistory(const std::vector<double>& history,
                                  const std::string& outputPath);

private:
    // Вспомогательные методы для генерации данных
    
    // Генерация сетки точек для построения границы решения
    static std::vector<std::pair<double, double>> generateGrid(
        const DataSet& data, 
        int gridSize = 100,
        double margin = 0.5);
    
    // Сохранение данных точек в CSV файл
    static void savePointsToCSV(const DataSet& data, 
                               const std::string& filename);
    
    // Сохранение границы решения в CSV файл
    static void saveDecisionBoundaryToCSV(
        const NeuralNetwork& net,
        const std::vector<std::pair<double, double>>& grid,
        const std::string& filename);
    
    // Генерация Python скрипта для визуализации
    static void generateVisualizationScript(
        const std::string& pointsFile,
        const std::string& boundaryFile,
        const std::string& outputImage);
    
    // Генерация Python скрипта для истории обучения
    static void generateTrainingHistoryScript(
        const std::string& historyFile,
        const std::string& outputImage);
};