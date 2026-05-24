#pragma once
#include <vector>
#include <string>
#include <map>

class NeuralNetwork;
class DataSet;

struct ClassMetrics {
    double precision;
    double recall;
    double f1;
    int truePositives;
    int falsePositives;
    int falseNegatives;
    int trueNegatives;
    int support; // Количество примеров этого класса
    
    ClassMetrics() 
        : precision(0.0), recall(0.0), f1(0.0), 
          truePositives(0), falsePositives(0), falseNegatives(0), trueNegatives(0), support(0) {}
};

struct ClassificationMetrics {
    double accuracy;
    double macroPrecision;
    double macroRecall;
    double macroF1;
    double weightedPrecision;
    double weightedRecall;
    double weightedF1;
    std::map<int, ClassMetrics> perClass;
    int totalSamples;
    
    ClassificationMetrics() 
        : accuracy(0.0), macroPrecision(0.0), macroRecall(0.0), macroF1(0.0),
          weightedPrecision(0.0), weightedRecall(0.0), weightedF1(0.0), totalSamples(0) {}
};

// Основная функция расчета метрик
ClassificationMetrics calculateMetrics(const NeuralNetwork& net, const DataSet& data);

// Вспомогательные функции
void printMetrics(const ClassificationMetrics& metrics);
double calculateFinalScore(double f1_d1, double f1_d2);
