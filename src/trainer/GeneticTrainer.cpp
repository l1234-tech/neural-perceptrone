#include "trainer/GeneticTrainer.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "metrics/ClassificationMetrics.h"
#include "utils/Random.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

GeneticTrainer::GeneticTrainer(size_t populationSize,
                               size_t maxGenerations,
                               double mutationRate,
                               double crossoverRate,
                               double elitismRatio,
                               double regularizationLambda,
                               size_t earlyStoppingPatience,
                               bool useF1Fitness)
    : populationSize(populationSize),
      maxGenerations(maxGenerations),
      mutationRate(mutationRate),
      crossoverRate(crossoverRate),
      elitismRatio(elitismRatio),
      regularizationLambda(regularizationLambda),
      earlyStoppingPatience(earlyStoppingPatience),
      useF1Fitness(useF1Fitness) {
    
    if (populationSize < 10) {
        throw std::invalid_argument("Размер популяции должен быть не менее 10");
    }
    if (elitismRatio < 0.0 || elitismRatio > 0.5) {
        throw std::invalid_argument("Доля элитных особей должна быть в диапазоне [0, 0.5]");
    }
    if (regularizationLambda < 0.0) {
        throw std::invalid_argument("Коэффициент регуляризации должен быть неотрицательным");
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
        
        // Вычисляем ошибку (loss) на обучающей выборке с регуляризацией L2
        individual.loss = network.computeLoss(trainingData, regularizationLambda);
        
        // ПРОСТАЯ fitness-функция: используем F1-score на валидации ИЛИ обратную ошибку
        if (useF1Fitness && !validationData.empty()) {
            // Используем F1-score на валидационной выборке как fitness
            ClassificationMetrics metrics = calculateMetrics(network, validationData);
            individual.f1Score = metrics.macroF1;
            individual.fitness = individual.f1Score + 1e-8; // Добавляем epsilon для избежания нуля
        } else {
            // Используем обратную ошибку как fitness (традиционный подход)
            individual.f1Score = 0.0;
            individual.fitness = 1.0 / (individual.loss + 1e-8);
        }
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
    validationLossHistory.clear();
    validationF1History.clear();
    
    // Сохраняем валидационную выборку для вычисления F1
    this->validationData = validationData;
    
    initializePopulation(network);
    
    size_t noImprovementCount = 0;
    double bestValidationF1 = 0.0;
    double bestValidationLoss = std::numeric_limits<double>::max();
    
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
                      << ", лучшая fitness = " << std::setprecision(4) << population[0].fitness;
            
            if (useF1Fitness && !validationData.empty() && population[0].f1Score > 0) {
                std::cout << ", F1 = " << std::setprecision(4) << population[0].f1Score;
            }
            std::cout << std::endl;
        }
        
        // Ранняя остановка по валидации (если есть валидационные данные)
        if (!validationData.empty()) {
            std::vector<double> originalWeights = network.getWeights();
            network.setWeights(population[0].chromosome);
            
            // Вычисляем ошибку на валидации
            double validationLoss = network.computeLoss(validationData, 0.0); // без регуляризации для оценки
            validationLossHistory.push_back(validationLoss);
            
            // Вычисляем F1-score на валидации
            ClassificationMetrics metrics = calculateMetrics(network, validationData);
            double validationF1 = metrics.macroF1;
            validationF1History.push_back(validationF1);
            
            network.setWeights(originalWeights);
            
            // Ранняя остановка по F1 (если используем F1 как fitness)
            if (useF1Fitness) {
                if (validationF1 > bestValidationF1) {
                    bestValidationF1 = validationF1;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            } else {
                // Ранняя остановка по loss
                if (validationLoss < bestValidationLoss) {
                    bestValidationLoss = validationLoss;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            }
            
            if (noImprovementCount >= earlyStoppingPatience) {
                std::cout << "Ранняя остановка: нет улучшений на валидации в течение " 
                          << earlyStoppingPatience << " поколений" << std::endl;
                break;
            }
        }
        
        tournamentSelection();
        uniformCrossover();
        simpleMutation();
    }
    
    // Устанавливаем лучшие веса в сеть
    network.setWeights(population[0].chromosome);
    
    // Оцениваем на валидационной выборке, если она предоставлена
    if (!validationData.empty() && validationLossHistory.empty()) {
        double validationLoss = network.computeLoss(validationData, 0.0);
        std::cout << "Ошибка на валидационной выборке: " << validationLoss << std::endl;
    }
    
    return bestLossHistory;
}

std::vector<double> GeneticTrainer::continueTraining(NeuralNetwork& network,
                                                    const DataSet& trainingData,
                                                    const DataSet& validationData) {
    std::cout << "Дообучение модели генетическим алгоритмом на новых данных...\n";
    
    // Сохраняем текущие истории и параметры
    std::vector<double> oldLossHistory = lossHistory;
    std::vector<double> oldBestLossHistory = bestLossHistory;
    std::vector<double> oldValidationLossHistory = validationLossHistory;
    std::vector<double> oldValidationF1History = validationF1History;
    double originalMutationRate = mutationRate;
    size_t originalMaxGenerations = maxGenerations;
    
    // Уменьшаем вероятность мутации для дообучения (ведь веса уже хорошие)
    double fineTuneMutationRate = originalMutationRate * 0.3;
    if (fineTuneMutationRate < 0.01) fineTuneMutationRate = 0.01;
    mutationRate = fineTuneMutationRate;
    
    // Уменьшаем количество поколений для дообучения
    size_t fineTuneGenerations = originalMaxGenerations / 2;
    if (fineTuneGenerations < 10) fineTuneGenerations = 10;
    maxGenerations = fineTuneGenerations;
    
    std::cout << "Параметры дообучения: mutationRate уменьшена с " << originalMutationRate
              << " до " << fineTuneMutationRate
              << ", maxGenerations уменьшено с " << originalMaxGenerations
              << " до " << fineTuneGenerations << "\n";
    
    // Обучаем на новых данных (история будет очищена в методе train)
    std::vector<double> newHistory = train(network, trainingData, validationData);
    
    // Восстанавливаем исходные параметры
    mutationRate = originalMutationRate;
    maxGenerations = originalMaxGenerations;
    
    // Объединяем истории
    lossHistory.insert(lossHistory.begin(), oldLossHistory.begin(), oldLossHistory.end());
    bestLossHistory.insert(bestLossHistory.begin(), oldBestLossHistory.begin(), oldBestLossHistory.end());
    validationLossHistory.insert(validationLossHistory.begin(), oldValidationLossHistory.begin(), oldValidationLossHistory.end());
    validationF1History.insert(validationF1History.begin(), oldValidationF1History.begin(), oldValidationF1History.end());
    
    // Объединяем возвращаемую историю
    std::vector<double> combinedHistory = oldBestLossHistory;
    combinedHistory.insert(combinedHistory.end(), newHistory.begin(), newHistory.end());
    
    std::cout << "Дообучение завершено. Общая история обучения теперь содержит " 
              << combinedHistory.size() << " поколений.\n";
    
    return combinedHistory;
}

std::string GeneticTrainer::getConfig() const {
    std::ostringstream ss;
    ss << "Генетический алгоритм (упрощенный):\n"
       << "  Размер популяции: " << populationSize << "\n"
       << "  Максимальное число поколений: " << maxGenerations << "\n"
       << "  Вероятность мутации: " << mutationRate << "\n"
       << "  Вероятность скрещивания: " << crossoverRate << "\n"
       << "  Доля элитных особей: " << elitismRatio << "\n"
       << "  Коэффициент регуляризации L2: " << regularizationLambda << "\n"
       << "  Терпение ранней остановки: " << earlyStoppingPatience << "\n"
       << "  Использовать F1 как fitness: " << (useF1Fitness ? "да" : "нет") << "\n"
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
         << elitismRatio << " " << regularizationLambda << " "
         << earlyStoppingPatience << " " << (useF1Fitness ? 1 : 0) << "\n";
    
    file << population.size() << "\n";
    for (const auto& ind : population) {
        file << ind.chromosome.size() << " " 
             << ind.fitness << " " << ind.loss << " " << ind.f1Score << "\n";
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
    
    file << validationLossHistory.size() << "\n";
    for (double loss : validationLossHistory) {
        file << loss << " ";
    }
    file << "\n";
    
    file << validationF1History.size() << "\n";
    for (double f1 : validationF1History) {
        file << f1 << " ";
    }
    file << "\n";
    
    file.close();
}

void GeneticTrainer::loadState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для загрузки: " + filename);
    }
    
    int useF1Int;
    file >> populationSize >> maxGenerations >> mutationRate 
         >> crossoverRate >> elitismRatio >> regularizationLambda
         >> earlyStoppingPatience >> useF1Int;
    useF1Fitness = (useF1Int != 0);
    
    size_t popSize;
    file >> popSize;
    
    population.clear();
    population.reserve(popSize);
    
    for (size_t i = 0; i < popSize; ++i) {
        size_t chromSize;
        double fitness, loss, f1Score;
        file >> chromSize >> fitness >> loss >> f1Score;
        
        GeneticIndividual ind(chromSize);
        ind.fitness = fitness;
        ind.loss = loss;
        ind.f1Score = f1Score;
        
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
    
    size_t validationLossHistorySize;
    file >> validationLossHistorySize;
    
    validationLossHistory.clear();
    validationLossHistory.reserve(validationLossHistorySize);
    for (size_t i = 0; i < validationLossHistorySize; ++i) {
        double loss;
        file >> loss;
        validationLossHistory.push_back(loss);
    }
    
    size_t validationF1HistorySize;
    file >> validationF1HistorySize;
    
    validationF1History.clear();
    validationF1History.reserve(validationF1HistorySize);
    for (size_t i = 0; i < validationF1HistorySize; ++i) {
        double f1;
        file >> f1;
        validationF1History.push_back(f1);
    }
    
    file.close();
}

std::vector<std::pair<std::string, double>> GeneticTrainer::getMetrics() const {
    std::vector<std::pair<std::string, double>> metrics;
    
    if (!population.empty()) {
        metrics.emplace_back("Лучшая приспособленность", population[0].fitness);
        metrics.emplace_back("Лучшая ошибка", population[0].loss);
        
        if (useF1Fitness && !validationData.empty()) {
            metrics.emplace_back("Лучший F1-score", population[0].f1Score);
        }
        
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
    
    if (!validationLossHistory.empty()) {
        metrics.emplace_back("Лучшая валидационная ошибка", *std::min_element(validationLossHistory.begin(), validationLossHistory.end()));
        metrics.emplace_back("Финальная валидационная ошибка", validationLossHistory.back());
    }
    
    if (!validationF1History.empty()) {
        metrics.emplace_back("Лучший валидационный F1", *std::max_element(validationF1History.begin(), validationF1History.end()));
        metrics.emplace_back("Финальный валидационный F1", validationF1History.back());
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

double GeneticTrainer::getBestF1Score() const {
    if (population.empty()) {
        return 0.0;
    }
    
    return population[0].f1Score;
}

void GeneticTrainer::setParameters(size_t popSize, size_t maxGen, double mutRate,
                                  double crossRate, double eliteRatio) {
    populationSize = popSize;
    maxGenerations = maxGen;
    mutationRate = mutRate;
    crossoverRate = crossRate;
    elitismRatio = eliteRatio;
}

ClassificationMetrics GeneticTrainer::evaluateMetrics(NeuralNetwork& network, const DataSet& data) const {
    if (population.empty()) {
        throw std::runtime_error("Популяция пуста");
    }
    
    // Сохраняем текущие веса
    std::vector<double> originalWeights = network.getWeights();
    
    // Устанавливаем лучшие веса
    network.setWeights(population[0].chromosome);
    
    // Вычисляем метрики
    ClassificationMetrics metrics = calculateMetrics(network, data);
    
    // Восстанавливаем оригинальные веса
    network.setWeights(originalWeights);
    
    return metrics;
}