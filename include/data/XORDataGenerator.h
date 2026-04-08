#pragma once
#include "data/DataSet.h"
#include "data/DataPoint.h"
#include "utils/Random.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>

class XORDataGenerator {
private:
    size_t trainSize;      // Размер обучающей выборки
    size_t testSize;       // Размер тестовой выборки
    double noiseLevel;     // Уровень шума в данных 
    double classRatio;     // Соотношение классов 
    int seed;              
    
public:
    XORDataGenerator(size_t trainSize = 300, 
                    size_t testSize = 100,
                    double noiseLevel = 0.1,
                    double classRatio = 0.6,
                    int seed = 42);
    
    std::pair<DataSet, DataSet> generateData();
    
    DataSet generateTrainData();
    
    DataSet generateTestData();
    
    void saveToCSV(const DataSet& trainData,
                  const DataSet& testData,
                  const std::string& trainPath = "data/train.csv",
                  const std::string& testPath = "data/test.csv");
    
    void setTrainSize(size_t size) { trainSize = size; }
    void setTestSize(size_t size) { testSize = size; }
    void setNoiseLevel(double level) { noiseLevel = std::max(0.0, std::min(1.0, level)); }
    void setClassRatio(double ratio) { classRatio = std::max(0.3, std::min(0.7, ratio)); }
    void setSeed(int newSeed) { seed = newSeed; }
    
    void printStats(const DataSet& trainData, const DataSet& testData) const;
    
private:
    DataSet generateDataset(size_t size, bool isTrain);
    std::pair<double, double> generatePoint(bool targetClass);
    int calculateLabel(double x, double y, bool addNoise);
    void addSpecialPoints(DataSet& dataset, size_t count);
};