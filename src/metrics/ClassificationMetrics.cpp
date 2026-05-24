#include "metrics/ClassificationMetrics.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "data/DataPoint.h"
#include "utils/Vector.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <set>

ClassificationMetrics calculateMetrics(const NeuralNetwork& net, const DataSet& data) {
    ClassificationMetrics metrics;
    
    if (data.empty()) {
        return metrics;
    }
    
    metrics.totalSamples = static_cast<int>(data.size());
    
    // Определяем уникальные классы
    std::set<int> uniqueClasses;
    for (size_t i = 0; i < data.size(); ++i) {
        uniqueClasses.insert(data.get(i).getLabel());
    }
    
    // Инициализируем метрики для каждого класса
    for (int cls : uniqueClasses) {
        metrics.perClass[cls] = ClassMetrics();
    }
    
    // Считаем предсказания
    int correct = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        const DataPoint& point = data.get(i);
        const auto& features = point.getFeatures();
        
        Vector input(features.size());
        for (size_t j = 0; j < features.size(); ++j) {
            input[j] = features[j];
        }
        
        int predicted = net.predict(input);
        int expected = point.getLabel();
        
        if (predicted == expected) {
            correct++;
        }
        
        // Обновляем confusion matrix для каждого класса
        for (int cls : uniqueClasses) {
            if (expected == cls && predicted == cls) {
                metrics.perClass[cls].truePositives++;
            } else if (expected != cls && predicted == cls) {
                metrics.perClass[cls].falsePositives++;
            } else if (expected == cls && predicted != cls) {
                metrics.perClass[cls].falseNegatives++;
            } else {
                metrics.perClass[cls].trueNegatives++;
            }
        }
        
        metrics.perClass[expected].support++;
    }
    
    metrics.accuracy = static_cast<double>(correct) / data.size();
    
    // Рассчитываем precision, recall, F1 для каждого класса
    double sumPrecision = 0.0;
    double sumRecall = 0.0;
    double sumF1 = 0.0;
    double weightedPrecision = 0.0;
    double weightedRecall = 0.0;
    double weightedF1 = 0.0;
    
    for (auto& [cls, classMetrics] : metrics.perClass) {
        // Precision = TP / (TP + FP)
        int tp_fp = classMetrics.truePositives + classMetrics.falsePositives;
        classMetrics.precision = tp_fp > 0 ? static_cast<double>(classMetrics.truePositives) / tp_fp : 0.0;
        
        // Recall = TP / (TP + FN)
        int tp_fn = classMetrics.truePositives + classMetrics.falseNegatives;
        classMetrics.recall = tp_fn > 0 ? static_cast<double>(classMetrics.truePositives) / tp_fn : 0.0;
        
        // F1 = 2 * (Precision * Recall) / (Precision + Recall)
        if (classMetrics.precision + classMetrics.recall > 0) {
            classMetrics.f1 = 2.0 * classMetrics.precision * classMetrics.recall / 
                             (classMetrics.precision + classMetrics.recall);
        } else {
            classMetrics.f1 = 0.0;
        }
        
        sumPrecision += classMetrics.precision;
        sumRecall += classMetrics.recall;
        sumF1 += classMetrics.f1;
        
        double weight = static_cast<double>(classMetrics.support) / data.size();
        weightedPrecision += classMetrics.precision * weight;
        weightedRecall += classMetrics.recall * weight;
        weightedF1 += classMetrics.f1 * weight;
    }
    
    // Macro-метрики (усреднение по классам)
    size_t numClasses = uniqueClasses.size();
    metrics.macroPrecision = sumPrecision / numClasses;
    metrics.macroRecall = sumRecall / numClasses;
    metrics.macroF1 = sumF1 / numClasses;
    
    // Weighted-метрики (взвешенное усреднение)
    metrics.weightedPrecision = weightedPrecision;
    metrics.weightedRecall = weightedRecall;
    metrics.weightedF1 = weightedF1;
    
    return metrics;
}

void printMetrics(const ClassificationMetrics& metrics) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  МЕТРИКИ КЛАССИФИКАЦИИ\n";
    std::cout << std::string(60, '=') << "\n";
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Общая точность (Accuracy): " << metrics.accuracy << "\n";
    std::cout << "Всего примеров: " << metrics.totalSamples << "\n\n";
    
    std::cout << "Метрики по классам:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(8) << "Класс" 
              << std::setw(12) << "Precision" 
              << std::setw(12) << "Recall" 
              << std::setw(12) << "F1-Score"
              << std::setw(10) << "Support" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& [cls, classMetrics] : metrics.perClass) {
        std::cout << std::setw(8) << cls 
                  << std::setw(12) << classMetrics.precision 
                  << std::setw(12) << classMetrics.recall 
                  << std::setw(12) << classMetrics.f1
                  << std::setw(10) << classMetrics.support << "\n";
    }
    
    std::cout << std::string(60, '-') << "\n";
    std::cout << "\nMacro-метрики (усреднение по классам):\n";
    std::cout << "  Macro Precision: " << metrics.macroPrecision << "\n";
    std::cout << "  Macro Recall:    " << metrics.macroRecall << "\n";
    std::cout << "  Macro F1-Score:  " << metrics.macroF1 << "\n";
    
    std::cout << "\nWeighted-метрики (взвешенное усреднение):\n";
    std::cout << "  Weighted Precision: " << metrics.weightedPrecision << "\n";
    std::cout << "  Weighted Recall:    " << metrics.weightedRecall << "\n";
    std::cout << "  Weighted F1-Score:  " << metrics.weightedF1 << "\n";
    
    std::cout << std::string(60, '=') << "\n";
}

double calculateFinalScore(double f1_d1, double f1_d2) {
    return 0.5 * f1_d1 + 0.5 * f1_d2;
}
