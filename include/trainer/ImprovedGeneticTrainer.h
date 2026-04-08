#pragma once
#include "trainer/ITrainer.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "utils/Random.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>

// Улучшенная особь с дополнительной информацией
struct ImprovedIndividual {
    std::vector<double> chromosome;  // Веса нейронной сети
    double fitness;                  // Приспособленность (1 / (loss + epsilon))
    double loss;                     // Ошибка на обучающей выборке
    int age;                         // Возраст особи (поколений)
    
    ImprovedIndividual(size_t chromosomeSize = 0) 
        : chromosome(chromosomeSize, 0.0), fitness(0.0), loss(0.0), age(0) {}
    
    // Оператор сравнения для сортировки (лучшие особи first)
    bool operator<(const ImprovedIndividual& other) const {
        return fitness > other.fitness; // Сортируем по убыванию приспособленности
    }
};

class ImprovedGeneticTrainer : public ITrainer {
private:
    // Параметры алгоритма (оптимизированы для XOR)
    size_t populationSize;      // Размер популяции
    size_t maxGenerations;      // Максимальное число поколений
    double mutationRate;        // Вероятность мутации
    double crossoverRate;       // Вероятность скрещивания
    double elitismRatio;        // Доля элитных особей
    double learningRate;        // Скорость обучения для локального поиска
    
    // Текущее состояние
    std::vector<ImprovedIndividual> population;
    std::vector<double> lossHistory;
    std::vector<double> bestLossHistory;
    std::vector<double> avgFitnessHistory;
    
    // Вспомогательные методы
    void initializePopulation(NeuralNetwork& network);
    void evaluatePopulation(NeuralNetwork& network, const DataSet& trainingData);
    void tournamentSelection(size_t tournamentSize = 3);
    void uniformCrossover();
    void adaptiveMutation();
    void applyGradientDescentStep(NeuralNetwork& network, ImprovedIndividual& individual, 
                                 const DataSet& trainingData);
    void agePopulation();
    
public:
    // Конструктор с оптимизированными параметрами для XOR
    ImprovedGeneticTrainer(size_t populationSize = 80,
                          size_t maxGenerations = 150,
                          double mutationRate = 0.15,
                          double crossoverRate = 0.7,
                          double elitismRatio = 0.15,
                          double learningRate = 0.05);
    
    // Обучить нейронную сеть (реализация интерфейса ITrainer)
    std::vector<double> train(NeuralNetwork& network, 
                             const DataSet& trainingData,
                             const DataSet& validationData = DataSet()) override;
    
    // Получить текущую конфигурацию
    std::string getConfig() const override;
    
    // Сохранить/загрузить состояние
    void saveState(const std::string& filename) const override;
    void loadState(const std::string& filename) override;
    
    // Получить метрики обучения
    std::vector<std::pair<std::string, double>> getMetrics() const override;
    
    // Дополнительные методы
    ImprovedIndividual getBestIndividual() const;
    double getBestLoss() const;
    double getAverageFitness() const;
    const std::vector<double>& getLossHistory() const { return lossHistory; }
    const std::vector<double>& getBestLossHistory() const { return bestLossHistory; }
    
    // Настройка параметров
    void setParameters(size_t popSize, size_t maxGen, double mutRate, 
                      double crossRate, double eliteRatio, double lr);
};