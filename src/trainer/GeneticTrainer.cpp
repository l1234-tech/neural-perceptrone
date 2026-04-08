#include "trainer/GeneticTrainer.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "utils/Random.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

GeneticTrainer::GeneticTrainer(size_t populationSize,
                               size_t maxGenerations,
                               double mutationRate,
                               double crossoverRate,
                               double elitismRatio)
    : populationSize(populationSize),
      maxGenerations(maxGenerations),
      mutationRate(mutationRate),
      crossoverRate(crossoverRate),
      elitismRatio(elitismRatio) {
    
    if (populationSize < 10) {
        throw std::invalid_argument("Размер популяции должен быть не менее 10");
    }
    if (elitismRatio < 0.0 || elitismRatio > 0.5) {
        throw std::invalid_argument("Доля элитных особей должна быть в диапазоне [0, 0.5]");
    }
}

void GeneticTrainer::initializePopulation(NeuralNetwork& network) {
    population.clear();
    population.reserve(populationSize);
    
    size_t chromosomeSize = network.getTotalParameters();
    
    for (size_t i = 0; i < populationSize; ++i) {
        GeneticIndividual individual(chromosomeSize);
        
        double range = 0.5;
        for (size_t j = 0; j < chromosomeSize; ++j) {
            individual.chromosome[j] = randomDouble(-range, range);
        }
        
        population.push_back(individual);
    }
}

void GeneticTrainer::evaluatePopulation(NeuralNetwork& network, const DataSet& trainingData) {
    std::vector<double> originalWeights = network.getWeights();
    
    for (auto& individual : population) {
   
        network.setWeights(individual.chromosome);
        
        // Вычисляем ошибку (loss) на обучающей выборке
        individual.loss = network.computeLoss(trainingData);

        individual.fitness = 1.0 / (individual.loss + 1e-8);
    }
    
    // Восстанавливаем оригинальные веса
    network.setWeights(originalWeights);
    
    // Сортируем популяцию по приспособленности 
    std::sort(population.begin(), population.end());
}

void GeneticTrainer::tournamentSelection(size_t tournamentSize) {
    std::vector<GeneticIndividual> newPopulation;
    newPopulation.reserve(populationSize);
    
    // Сохраняем элитных особей
    size_t eliteCount = static_cast<size_t>(populationSize * elitismRatio);
    for (size_t i = 0; i < eliteCount && i < population.size(); ++i) {
        newPopulation.push_back(population[i]);
    }
    
    while (newPopulation.size() < populationSize) {
        std::vector<size_t> tournamentIndices;
        for (size_t i = 0; i < tournamentSize; ++i) {
            tournamentIndices.push_back(randomInt(0, population.size() - 1));
        }
        
        size_t bestIndex = tournamentIndices[0];
        double bestFitness = population[bestIndex].fitness;
        
        for (size_t i = 1; i < tournamentSize; ++i) {
            size_t idx = tournamentIndices[i];
            if (population[idx].fitness > bestFitness) {
                bestIndex = idx;
                bestFitness = population[idx].fitness;
            }
        }
        
        newPopulation.push_back(population[bestIndex]);
    }
    
    population = std::move(newPopulation);
}

void GeneticTrainer::uniformCrossover() {
    for (size_t i = 0; i + 1 < population.size(); i += 2) {
        if (randomDouble(0.0, 1.0) < crossoverRate) {
            GeneticIndividual& parent1 = population[i];
            GeneticIndividual& parent2 = population[i + 1];
            
            for (size_t j = 0; j < parent1.chromosome.size(); ++j) {
                if (randomDouble(0.0, 1.0) < 0.5) {
                    std::swap(parent1.chromosome[j], parent2.chromosome[j]);
                }
            }
        }
    }
}

void GeneticTrainer::simpleMutation() {
    for (size_t i = 0; i < population.size(); ++i) {
        for (size_t j = 0; j < population[i].chromosome.size(); ++j) {
            if (randomDouble(0.0, 1.0) < mutationRate) {

                double currentValue = population[i].chromosome[j];
                double mutationScale = 0.1 * (1.0 + std::abs(currentValue));
                double mutation = randomDouble(-mutationScale, mutationScale);
                population[i].chromosome[j] += mutation;
                
                // Ограничиваем значения весов
                if (population[i].chromosome[j] > 5.0) population[i].chromosome[j] = 5.0;
                if (population[i].chromosome[j] < -5.0) population[i].chromosome[j] = -5.0;
            }
        }
    }
}

std::vector<double> GeneticTrainer::train(NeuralNetwork& network,
                                         const DataSet& trainingData,
                                         const DataSet& validationData) {
    if (trainingData.empty()) {
        throw std::invalid_argument("Обучающая выборка не может быть пустой");
    }
    
    lossHistory.clear();
    bestLossHistory.clear();
    
    initializePopulation(network);
    
    for (size_t generation = 0; generation < maxGenerations; ++generation) {
        // Оценка приспособленности
        evaluatePopulation(network, trainingData);
        
        // Сохраняем статистику
        double bestLoss = population[0].loss;
        double avgLoss = 0.0;
        
        for (const auto& ind : population) {
            avgLoss += ind.loss;
        }
        
        avgLoss /= population.size();
        
        lossHistory.push_back(avgLoss);
        bestLossHistory.push_back(bestLoss);
        
        // Вывод прогресса каждые 20 поколений
        if (generation % 20 == 0) {
            std::cout << "Поколение " << std::setw(3) << generation 
                      << ": лучшая ошибка = " << std::fixed << std::setprecision(6) << bestLoss
                      << ", средняя ошибка = " << avgLoss 
                      << ", лучшая fitness = " << std::setprecision(4) << population[0].fitness << std::endl;
        }
        
        tournamentSelection();
        
        uniformCrossover();
        
        simpleMutation();

        if (bestLoss < 0.05) {
            std::cout << "Ранняя остановка: достигнута достаточная точность (loss < 0.05)" << std::endl;
            break;
        }
    }
    
    // Устанавливаем лучшие веса в сеть
    network.setWeights(population[0].chromosome);
    
    // Оцениваем на валидационной выборке, если она предоставлена
    if (!validationData.empty()) {
        double validationLoss = network.computeLoss(validationData);
        std::cout << "Ошибка на валидационной выборке: " << validationLoss << std::endl;
    }
    
    return bestLossHistory;
}

std::string GeneticTrainer::getConfig() const {
    std::ostringstream ss;
    ss << "Генетический алгоритм (упрощенный):\n"
       << "  Размер популяции: " << populationSize << "\n"
       << "  Максимальное число поколений: " << maxGenerations << "\n"
       << "  Вероятность мутации: " << mutationRate << "\n"
       << "  Вероятность скрещивания: " << crossoverRate << "\n"
       << "  Доля элитных особей: " << elitismRatio << "\n"
       << "  Текущий размер популяции: " << population.size() << "\n"
       << "  История ошибок: " << lossHistory.size() << " записей";
    
    return ss.str();
}

void GeneticTrainer::saveState(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для сохранения: " + filename);
    }
    
    file << populationSize << " " << maxGenerations << " "
         << mutationRate << " " << crossoverRate << " " 
         << elitismRatio << "\n";
    
    file << population.size() << "\n";
    for (const auto& ind : population) {
        file << ind.chromosome.size() << " " 
             << ind.fitness << " " << ind.loss << "\n";
        for (double gene : ind.chromosome) {
            file << gene << " ";
        }
        file << "\n";
    }
    
    file << lossHistory.size() << "\n";
    for (double loss : lossHistory) {
        file << loss << " ";
    }
    file << "\n";
    
    file << bestLossHistory.size() << "\n";
    for (double loss : bestLossHistory) {
        file << loss << " ";
    }
    file << "\n";
    
    file.close();
}

void GeneticTrainer::loadState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для загрузки: " + filename);
    }
    
    file >> populationSize >> maxGenerations >> mutationRate 
         >> crossoverRate >> elitismRatio;
    
    size_t popSize;
    file >> popSize;
    
    population.clear();
    population.reserve(popSize);
    
    for (size_t i = 0; i < popSize; ++i) {
        size_t chromSize;
        double fitness, loss;
        file >> chromSize >> fitness >> loss;
        
        GeneticIndividual ind(chromSize);
        ind.fitness = fitness;
        ind.loss = loss;
        
        for (size_t j = 0; j < chromSize; ++j) {
            file >> ind.chromosome[j];
        }
        
        population.push_back(ind);
    }
    
    size_t lossHistorySize;
    file >> lossHistorySize;
    
    lossHistory.clear();
    lossHistory.reserve(lossHistorySize);
    for (size_t i = 0; i < lossHistorySize; ++i) {
        double loss;
        file >> loss;
        lossHistory.push_back(loss);
    }
    
    size_t bestLossHistorySize;
    file >> bestLossHistorySize;
    
    bestLossHistory.clear();
    bestLossHistory.reserve(bestLossHistorySize);
    for (size_t i = 0; i < bestLossHistorySize; ++i) {
        double loss;
        file >> loss;
        bestLossHistory.push_back(loss);
    }
    
    file.close();
}

std::vector<std::pair<std::string, double>> GeneticTrainer::getMetrics() const {
    std::vector<std::pair<std::string, double>> metrics;
    
    if (!population.empty()) {
        metrics.emplace_back("Лучшая приспособленность", population[0].fitness);
        metrics.emplace_back("Лучшая ошибка", population[0].loss);
        
        double avgFitness = 0.0;
        for (const auto& ind : population) {
            avgFitness += ind.fitness;
        }
        avgFitness /= population.size();
        metrics.emplace_back("Средняя приспособленность", avgFitness);
    }
    
    if (!lossHistory.empty()) {
        metrics.emplace_back("Начальная ошибка", lossHistory[0]);
        metrics.emplace_back("Финальная ошибка", lossHistory.back());
        
        if (lossHistory.size() > 1) {
            double improvement = (lossHistory[0] - lossHistory.back()) / lossHistory[0] * 100.0;
            metrics.emplace_back("Улучшение ошибки (%)", improvement);
        }
    }
    
    metrics.emplace_back("Размер популяции", static_cast<double>(population.size()));
    metrics.emplace_back("Поколений выполнено", static_cast<double>(lossHistory.size()));
    
    return metrics;
}

GeneticIndividual GeneticTrainer::getBestIndividual() const {
    if (population.empty()) {
        throw std::runtime_error("Популяция пуста");
    }
    
    return population[0];
}

double GeneticTrainer::getBestLoss() const {
    if (population.empty()) {
        return 0.0;
    }
    
    return population[0].loss;
}

void GeneticTrainer::setParameters(size_t popSize, size_t maxGen, double mutRate,
                                  double crossRate, double eliteRatio) {
    populationSize = popSize;
    maxGenerations = maxGen;
    mutationRate = mutRate;
    crossoverRate = crossRate;
    elitismRatio = eliteRatio;
}