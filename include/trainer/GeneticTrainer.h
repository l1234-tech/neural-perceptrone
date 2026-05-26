#pragma once
#include "trainer/ITrainer.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "metrics/ClassificationMetrics.h"
#include "utils/Random.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
struct GeneticIndividual {
    std::vector<double> chromosome;  // Веса нейронной сети
    double fitness;                  // Приспособленность (F1-score или 1/(loss+epsilon))
    double loss;                     // Ошибка на обучающей выборке
    double f1Score;                  // F1-score на валидационной выборке (если есть)
    
    GeneticIndividual(size_t chromosomeSize = 0) 
        : chromosome(chromosomeSize, 0.0), fitness(0.0), loss(0.0), f1Score(0.0) {}
    
    bool operator<(const GeneticIndividual& other) const {
        return fitness > other.fitness; // Сортируем по убыванию приспособленности
    }
};

class GeneticTrainer : public ITrainer {
private:
    // Параметры алгоритма
    size_t populationSize;      // Размер популяции
    size_t maxGenerations;      // Максимальное число поколений
    double mutationRate;        // Вероятность мутации
    double crossoverRate;       // Вероятность скрещивания
    double elitismRatio;        // Доля элитных особей
    double regularizationLambda; // Коэффициент L2 регуляризации
    size_t earlyStoppingPatience; // Количество поколений без улучшения для ранней остановки
    bool useF1Fitness;          // Использовать F1-score вместо 1/loss для fitness
    
    // Текущее состояние
    std::vector<GeneticIndividual> population;
    std::vector<double> lossHistory;
    std::vector<double> bestLossHistory;
    std::vector<double> validationLossHistory; // История ошибок на валидационной выборке
    std::vector<double> validationF1History;   // История F1-score на валидационной выборке
    DataSet validationData;                    // Валидационная выборка для вычисления F1
    
    void initializePopulation(NeuralNetwork& network);
    void evaluatePopulation(NeuralNetwork& network, const DataSet& trainingData);
    void tournamentSelection(size_t tournamentSize = 3);
    void uniformCrossover();
    void simpleMutation();
    
public:
    // Конструктор с параметрами по умолчанию
    GeneticTrainer(size_t populationSize = 50,
                   size_t maxGenerations = 100,
                   double mutationRate = 0.1,
                   double crossoverRate = 0.8,
                   double elitismRatio = 0.1,
                   double regularizationLambda = 0.01,
                   size_t earlyStoppingPatience = 10,
                   bool useF1Fitness = true);
    
    std::vector<double> train(NeuralNetwork& network, 
                             const DataSet& trainingData,
                             const DataSet& validationData = DataSet()) override;
    
    // Дообучение на новых данных с уменьшенной вероятностью мутации
    std::vector<double> continueTraining(NeuralNetwork& network, 
                                         const DataSet& trainingData,
                                         const DataSet& validationData = DataSet()) override;
    
    std::string getConfig() const override;

    void saveState(const std::string& filename) const override;
    void loadState(const std::string& filename) override;

    std::vector<std::pair<std::string, double>> getMetrics() const override;

    GeneticIndividual getBestIndividual() const;
    double getBestLoss() const;
    double getBestF1Score() const;
    const std::vector<double>& getLossHistory() const { return lossHistory; }
    const std::vector<double>& getBestLossHistory() const { return bestLossHistory; }
    const std::vector<double>& getValidationLossHistory() const { return validationLossHistory; }
    const std::vector<double>& getValidationF1History() const { return validationF1History; }

    void setParameters(size_t popSize, size_t maxGen, double mutRate, 
                      double crossRate, double eliteRatio);
    
    void setRegularizationLambda(double lambda) { regularizationLambda = lambda; }
    void setEarlyStoppingPatience(size_t patience) { earlyStoppingPatience = patience; }
    void setUseF1Fitness(bool useF1) { useF1Fitness = useF1; }
    void setValidationData(const DataSet& data) { validationData = data; }
    
    // Новые методы для работы с F1-метриками
    ClassificationMetrics evaluateMetrics(NeuralNetwork& network, const DataSet& data) const;
};