#include "utils/Visualizer.h"
#include "data/DataSet.h"
#include "neural_network/NeuralNetwork.h"
#include "utils/Vector.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <iostream>
#include <filesystem>

void Visualizer::saveDatasetToCSV(const DataSet& dataset, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи" << std::endl;
        return;
    }
    
    // Заголовок CSV в формате x1,x2,label (совместимость с существующими данными)
    file << "x1,x2,label" << std::endl;
    
    // Данные
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& point = dataset.get(i);
        const auto& features = point.getFeatures();
        
        if (features.size() >= 2) {
            file << std::fixed << std::setprecision(6) 
                 << features[0] << "," 
                 << features[1] << "," 
                 << point.getLabel() << std::endl;
        }
    }
    
    file.close();
    std::cout << "Датасет сохранен в " << filename << " (" << dataset.size() << " точек)" << std::endl;
}

void Visualizer::savePredictionsToCSV(const NeuralNetwork& network, 
                                     const DataSet& dataset, 
                                     const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи" << std::endl;
        return;
    }
    
    // Заголовок CSV
    file << "x1,x2,true_label,predicted_label,confidence" << std::endl;
    
    int correct = 0;
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& point = dataset.get(i);
        const auto& features = point.getFeatures();
        
        if (features.size() >= 2) {
            Vector input(features.size());
            for (size_t j = 0; j < features.size(); ++j) {
                input[j] = features[j];
            }
            
            int predicted = network.predict(input);
            Vector output = network.forward(input);
            double confidence = std::abs(output[0] - 0.5) * 2.0 * 100.0;
            
            if (predicted == point.getLabel()) {
                correct++;
            }
            
            file << std::fixed << std::setprecision(6) 
                 << features[0] << "," 
                 << features[1] << "," 
                 << point.getLabel() << ","
                 << predicted << ","
                 << confidence << std::endl;
        }
    }
    
    file.close();
    double accuracy = static_cast<double>(correct) / dataset.size() * 100.0;
    std::cout << "Предсказания сохранены в " << filename 
              << " (точность: " << std::fixed << std::setprecision(2) << accuracy << "%)" << std::endl;
}

void Visualizer::saveTrainingHistoryToCSV(const std::vector<double>& lossHistory,
                                         const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи" << std::endl;
        return;
    }
    
    // Заголовок CSV
    file << "generation,loss" << std::endl;
    
    for (size_t i = 0; i < lossHistory.size(); ++i) {
        file << i << "," << std::fixed << std::setprecision(6) << lossHistory[i] << std::endl;
    }
    
    file.close();
    std::cout << "История обучения сохранена в " << filename 
              << " (" << lossHistory.size() << " поколений)" << std::endl;
}

void Visualizer::saveDecisionBoundary(const NeuralNetwork& network,
                                     double xMin, double xMax,
                                     double yMin, double yMax,
                                     int gridSize,
                                     const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи" << std::endl;
        return;
    }
    
    // Заголовок CSV
    file << "x1,x2,prediction" << std::endl;
    
    double xStep = (xMax - xMin) / (gridSize - 1);
    double yStep = (yMax - yMin) / (gridSize - 1);
    
    for (int i = 0; i < gridSize; ++i) {
        double x = xMin + i * xStep;
        for (int j = 0; j < gridSize; ++j) {
            double y = yMin + j * yStep;
            
            Vector input(2);
            input[0] = x;
            input[1] = y;
            
            int prediction = network.predict(input);
            
            file << std::fixed << std::setprecision(6) 
                 << x << "," << y << "," << prediction << std::endl;
        }
    }
    
    file.close();
    std::cout << "Граница решения сохранена в " << filename 
              << " (" << gridSize << "x" << gridSize << " точек)" << std::endl;
}

void Visualizer::saveTestReport(const NeuralNetwork& network,
                               const DataSet& testData,
                               const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи" << std::endl;
        return;
    }
    
    file << "=== ОТЧЕТ О ТЕСТИРОВАНИИ НЕЙРОННОЙ СЕТИ ===" << std::endl;
    file << "Время: " << getCurrentTimestamp() << std::endl;
    file << std::endl;
    
    // Архитектура сети
    const auto& arch = network.getArchitecture();
    file << "Архитектура сети: [";
    for (size_t i = 0; i < arch.size(); ++i) {
        file << arch[i];
        if (i < arch.size() - 1) file << " → ";
    }
    file << "]" << std::endl;
    file << "Всего параметров: " << network.getTotalParameters() << std::endl;
    file << std::endl;
    
    // Статистика тестирования
    int correct = 0;
    int truePositives = 0, trueNegatives = 0, falsePositives = 0, falseNegatives = 0;
    
    for (size_t i = 0; i < testData.size(); ++i) {
        const auto& point = testData.get(i);
        const auto& features = point.getFeatures();
        
        Vector input(features.size());
        for (size_t j = 0; j < features.size(); ++j) {
            input[j] = features[j];
        }
        
        int predicted = network.predict(input);
        int expected = point.getLabel();
        
        if (predicted == expected) {
            correct++;
            if (expected == 0) trueNegatives++;
            else truePositives++;
        } else {
            if (expected == 0 && predicted == 1) falsePositives++;
            if (expected == 1 && predicted == 0) falseNegatives++;
        }
    }
    
    double accuracy = static_cast<double>(correct) / testData.size() * 100.0;
    double loss = network.computeLoss(testData);
    
    file << "=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===" << std::endl;
    file << "Размер тестовой выборки: " << testData.size() << std::endl;
    file << "Правильных предсказаний: " << correct << std::endl;
    file << "Точность: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    file << "Ошибка (loss): " << std::fixed << std::setprecision(6) << loss << std::endl;
    file << std::endl;
    
    // Расширенная статистика
    file << "=== РАСШИРЕННАЯ СТАТИСТИКА ===" << std::endl;
    file << "True Positives (TP): " << truePositives << std::endl;
    file << "True Negatives (TN): " << trueNegatives << std::endl;
    file << "False Positives (FP): " << falsePositives << std::endl;
    file << "False Negatives (FN): " << falseNegatives << std::endl;
    
    if (truePositives + falsePositives > 0) {
        double precision = static_cast<double>(truePositives) / (truePositives + falsePositives) * 100.0;
        file << "Precision: " << std::fixed << std::setprecision(2) << precision << "%" << std::endl;
    }
    
    if (truePositives + falseNegatives > 0) {
        double recall = static_cast<double>(truePositives) / (truePositives + falseNegatives) * 100.0;
        file << "Recall: " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
    }
    
    if (truePositives + falsePositives > 0 && truePositives + falseNegatives > 0) {
        double precision = static_cast<double>(truePositives) / (truePositives + falsePositives);
        double recall = static_cast<double>(truePositives) / (truePositives + falseNegatives);
        double f1Score = 2.0 * precision * recall / (precision + recall) * 100.0;
        file << "F1-Score: " << std::fixed << std::setprecision(2) << f1Score << "%" << std::endl;
    }
    
    file.close();
    std::cout << "Отчет о тестировании сохранен в " << filename << std::endl;
}

std::string Visualizer::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}