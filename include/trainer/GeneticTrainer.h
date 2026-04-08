#pragma once
#include "trainer/ITrainer.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "utils/Random.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
struct GeneticIndividual {
    std::vector<double> chromosome;  // Веса нейронной сети
    double fitness;                  // Приспособленность (1 / (loss + epsilon))
    double loss;                     // Ошибка на обучающей выборке
    
    GeneticIndividual(size_t chromosomeSize = 0) 
        : chromosome(chromosomeSize, 0.0), fitness(0.0), loss(0.0) {}
    
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
    
    // Текущее состояние
    std::vector<GeneticIndividual> population;
    std::vector<double> lossHistory;
    std::vector<double> bestLossHistory;
    
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
                   double elitismRatio = 0.1);
    
    std::vector<double> train(NeuralNetwork& network, 
                             const DataSet& trainingData,
                             const DataSet& validationData = DataSet()) override;
    
    std::string getConfig() const override;

    void saveState(const std::string& filename) const override;
    void loadState(const std::string& filename) override;

    std::vector<std::pair<std::string, double>> getMetrics() const override;

    GeneticIndividual getBestIndividual() const;
    double getBestLoss() const;
    const std::vector<double>& getLossHistory() const { return lossHistory; }
    const std::vector<double>& getBestLossHistory() const { return bestLossHistory; }

    void setParameters(size_t popSize, size_t maxGen, double mutRate, 
                      double crossRate, double eliteRatio);
};