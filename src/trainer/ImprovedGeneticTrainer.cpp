#include "trainer/ImprovedGeneticTrainer.h"
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

ImprovedGeneticTrainer::ImprovedGeneticTrainer(size_t populationSize,
                                               size_t maxGenerations,
                                               double mutationRate,
                                               double crossoverRate,
                                               double elitismRatio,
                                               double learningRate)
    : populationSize(populationSize),
      maxGenerations(maxGenerations),
      mutationRate(mutationRate),
      crossoverRate(crossoverRate),
      elitismRatio(elitismRatio),
      learningRate(learningRate) {
    
    if (populationSize < 10) {
        throw std::invalid_argument("Размер популяции должен быть не менее 10");
    }
    if (elitismRatio < 0.0 || elitismRatio > 0.5) {
        throw std::invalid_argument("Доля элитных особей должна быть в диапазоне [0, 0.5]");
    }
}

void ImprovedGeneticTrainer::initializePopulation(NeuralNetwork& network) {
    population.clear();
    population.reserve(populationSize);
    
    size_t chromosomeSize = network.getTotalParameters();
    
    // Инициализация популяции с разными стратегиями
    for (size_t i = 0; i < populationSize; ++i) {
        ImprovedIndividual individual(chromosomeSize);
        
        // Разные стратегии инициализации для разнообразия
        if (i < populationSize / 3) {
            // Маленькие случайные веса
            for (size_t j = 0; j < chromosomeSize; ++j) {
                individual.chromosome[j] = randomDouble(-0.5, 0.5);
            }
        } else if (i < 2 * populationSize / 3) {
            // Веса в диапазоне [-1, 1]
            for (size_t j = 0; j < chromosomeSize; ++j) {
                individual.chromosome[j] = randomDouble(-1.0, 1.0);
            }
        } else {
            // Очень маленькие веса (почти нулевые)
            for (size_t j = 0; j < chromosomeSize; ++j) {
                individual.chromosome[j] = randomDouble(-0.1, 0.1);
            }
        }
        
        population.push_back(individual);
    }
}

void ImprovedGeneticTrainer::evaluatePopulation(NeuralNetwork& network, const DataSet& trainingData) {
    // Сохраняем оригинальные веса сети
    std::vector<double> originalWeights = network.getWeights();
    
    for (auto& individual : population) {
        // Устанавливаем веса сети из хромосомы особи
        network.setWeights(individual.chromosome);
        
        // Вычисляем ошибку (loss) на обучающей выборке
        individual.loss = network.computeLoss(trainingData);
        
        // Приспособленность обратно пропорциональна ошибке
        // Добавляем маленькое значение, чтобы избежать деления на ноль
        individual.fitness = 1.0 / (individual.loss + 1e-8);
        
        // Увеличиваем возраст особи
        individual.age++;
    }
    
    // Восстанавливаем оригинальные веса
    network.setWeights(originalWeights);
    
    // Сортируем популяцию по приспособленности (лучшие first)
    std::sort(population.begin(), population.end());
}

void ImprovedGeneticTrainer::tournamentSelection(size_t tournamentSize) {
    std::vector<ImprovedIndividual> newPopulation;
    newPopulation.reserve(populationSize);
    
    // Сохраняем элитных особей
    size_t eliteCount = static_cast<size_t>(populationSize * elitismRatio);
    for (size_t i = 0; i < eliteCount && i < population.size(); ++i) {
        newPopulation.push_back(population[i]);
        newPopulation.back().age = population[i].age; // Сохраняем возраст
    }
    
    // Турнирная селекция для остальной части популяции
    while (newPopulation.size() < populationSize) {
        // Выбираем случайных особей для турнира
        std::vector<size_t> tournamentIndices;
        for (size_t i = 0; i < tournamentSize; ++i) {
            tournamentIndices.push_back(randomInt(0, population.size() - 1));
        }
        
        // Находим лучшую особь в турнире
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

void ImprovedGeneticTrainer::uniformCrossover() {
    // Однородный кроссинговер (каждый ген независимо от родителя)
    for (size_t i = 0; i + 1 < population.size(); i += 2) {
        if (randomDouble(0.0, 1.0) < crossoverRate) {
            ImprovedIndividual& parent1 = population[i];
            ImprovedIndividual& parent2 = population[i + 1];
            
            // Для каждого гена выбираем от какого родителя взять
            for (size_t j = 0; j < parent1.chromosome.size(); ++j) {
                if (randomDouble(0.0, 1.0) < 0.5) {
                    std::swap(parent1.chromosome[j], parent2.chromosome[j]);
                }
            }
        }
    }
}

void ImprovedGeneticTrainer::adaptiveMutation() {
    // Адаптивная мутация: больше мутаций для худших особей
    for (size_t i = 0; i < population.size(); ++i) {
        double individualMutationRate = mutationRate;
        
        // Увеличиваем вероятность мутации для худших особей
        if (i > population.size() / 2) {
            individualMutationRate *= 2.0;
        }
        
        // Уменьшаем вероятность мутации для старых хороших особей
        if (i < population.size() / 4 && population[i].age > 10) {
            individualMutationRate *= 0.5;
        }
        
        for (size_t j = 0; j < population[i].chromosome.size(); ++j) {
            if (randomDouble(0.0, 1.0) < individualMutationRate) {
                // Гауссова мутация с уменьшающимся размахом
                double mutationStrength = 0.5 * exp(-population[i].age / 50.0);
                double mutation = randomDouble(-mutationStrength, mutationStrength);
                population[i].chromosome[j] += mutation;
                
                // Ограничиваем значения весов
                if (population[i].chromosome[j] > 2.0) population[i].chromosome[j] = 2.0;
                if (population[i].chromosome[j] < -2.0) population[i].chromosome[j] = -2.0;
            }
        }
    }
}

void ImprovedGeneticTrainer::applyGradientDescentStep(NeuralNetwork& network, 
                                                     ImprovedIndividual& individual,
                                                     const DataSet& trainingData) {
    // Применяем только к лучшим особям (первые 20%)
    static size_t applyCount = populationSize / 5;
    
    // Сохраняем оригинальные веса
    std::vector<double> originalWeights = network.getWeights();
    
    // Устанавливаем веса из особи
    network.setWeights(individual.chromosome);
    
    // Выбираем случайный пример для градиентного шага
    if (!trainingData.empty()) {
        size_t randomIndex = randomInt(0, trainingData.size() - 1);
        const auto& point = trainingData.get(randomIndex);
        const auto& features = point.getFeatures();
        
        Vector input(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            input[i] = features[i];
        }
        
        Vector target(1, static_cast<double>(point.getLabel()));
        
        // Один шаг градиентного спуска
        network.trainOneExample(input, target, learningRate);
        
        // Сохраняем улучшенные веса
        individual.chromosome = network.getWeights();
    }
    
    // Восстанавливаем оригинальные веса
    network.setWeights(originalWeights);
}

void ImprovedGeneticTrainer::agePopulation() {
    // Увеличиваем возраст всех особей
    for (auto& individual : population) {
        individual.age++;
    }
}

std::vector<double> ImprovedGeneticTrainer::train(NeuralNetwork& network,
                                                 const DataSet& trainingData,
                                                 const DataSet& validationData) {
    if (trainingData.empty()) {
        throw std::invalid_argument("Обучающая выборка не может быть пустой");
    }
    
    lossHistory.clear();
    bestLossHistory.clear();
    avgFitnessHistory.clear();
    
    // Инициализируем популяцию
    initializePopulation(network);
    
    // Основной цикл генетического алгоритма
    for (size_t generation = 0; generation < maxGenerations; ++generation) {
        // Оценка приспособленности
        evaluatePopulation(network, trainingData);
        
        // Сохраняем статистику
        double bestLoss = population[0].loss;
        double avgLoss = 0.0;
        double avgFitness = 0.0;
        
        for (const auto& ind : population) {
            avgLoss += ind.loss;
            avgFitness += ind.fitness;
        }
        
        avgLoss /= population.size();
        avgFitness /= population.size();
        
        lossHistory.push_back(avgLoss);
        bestLossHistory.push_back(bestLoss);
        avgFitnessHistory.push_back(avgFitness);
        
        // Вывод прогресса каждые 20 поколений
        if (generation % 20 == 0) {
            std::cout << "Поколение " << std::setw(3) << generation 
                      << ": лучшая ошибка = " << std::fixed << std::setprecision(6) << bestLoss
                      << ", средняя ошибка = " << avgLoss
                      << ", средняя приспособленность = " << std::setprecision(4) << avgFitness << std::endl;
        }
        
        // Селекция
        tournamentSelection();
        
        // Скрещивание
        uniformCrossover();
        
        // Мутация
        adaptiveMutation();
        
        // Локальный поиск для лучших особей (градиентный спуск)
        size_t localSearchCount = std::min<size_t>(5, populationSize / 10);
        for (size_t i = 0; i < localSearchCount; ++i) {
            applyGradientDescentStep(network, population[i], trainingData);
        }
        
        // Старение популяции
        agePopulation();
        
        // Ранняя остановка, если ошибка достаточно мала
        if (bestLoss < 0.01) {
            std::cout << "Ранняя остановка: достигнута достаточная точность" << std::endl;
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

std::string ImprovedGeneticTrainer::getConfig() const {
    std::ostringstream ss;
    ss << "Улучшенный генетический алгоритм:\n"
       << "  Размер популяции: " << populationSize << "\n"
       << "  Максимальное число поколений: " << maxGenerations << "\n"
       << "  Вероятность мутации: " << mutationRate << "\n"
       << "  Вероятность скрещивания: " << crossoverRate << "\n"
       << "  Доля элитных особей: " << elitismRatio << "\n"
       << "  Скорость обучения (локальный поиск): " << learningRate << "\n"
       << "  Текущий размер популяции: " << population.size() << "\n"
       << "  История ошибок: " << lossHistory.size() << " записей";
    
    return ss.str();
}

void ImprovedGeneticTrainer::saveState(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для сохранения: " + filename);
    }
    
    // Сохраняем параметры
    file << populationSize << " " << maxGenerations << " "
         << mutationRate << " " << crossoverRate << " " 
         << elitismRatio << " " << learningRate << "\n";
    
    // Сохраняем популяцию
    file << population.size() << "\n";
    for (const auto& ind : population) {
        file << ind.chromosome.size() << " " 
             << ind.fitness << " " << ind.loss << " " << ind.age << "\n";
        for (double gene : ind.chromosome) {
            file << gene << " ";
        }
        file << "\n";
    }
    
    // Сохраняем историю
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
    
    file << avgFitnessHistory.size() << "\n";
    for (double fitness : avgFitnessHistory) {
        file << fitness << " ";
    }
    file << "\n";
    
    file.close();
}

void ImprovedGeneticTrainer::loadState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для загрузки: " + filename);
    }
    
    // Загружаем параметры
    file >> populationSize >> maxGenerations >> mutationRate 
         >> crossoverRate >> elitismRatio >> learningRate;
    
    // Загружаем популяцию
    size_t popSize;
    file >> popSize;
    
    population.clear();
    population.reserve(popSize);
    
    for (size_t i = 0; i < popSize; ++i) {
        size_t chromSize;
        double fitness, loss;
        int age;
        file >> chromSize >> fitness >> loss >> age;
        
        ImprovedIndividual ind(chromSize);
        ind.fitness = fitness;
        ind.loss = loss;
        ind.age = age;
        
        for (size_t j = 0; j < chromSize; ++j) {
            file >> ind.chromosome[j];
        }
        
        population.push_back(ind);
    }
    
    // Загружаем историю
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
    
    size_t avgFitnessHistorySize;
    file >> avgFitnessHistorySize;
    
    avgFitnessHistory.clear();
    avgFitnessHistory.reserve(avgFitnessHistorySize);
    for (size_t i = 0; i < avgFitnessHistorySize; ++i) {
        double fitness;
        file >> fitness;
        avgFitnessHistory.push_back(fitness);
    }
    
    file.close();
}

std::vector<std::pair<std::string, double>> ImprovedGeneticTrainer::getMetrics() const {
    std::vector<std::pair<std::string, double>> metrics;
    
    if (!population.empty()) {
        metrics.emplace_back("Лучшая приспособленность", population[0].fitness);
        metrics.emplace_back("Лучшая ошибка", population[0].loss);
        metrics.emplace_back("Средняя приспособленность", getAverageFitness());
        metrics.emplace_back("Возраст лучшей особи", static_cast<double>(population[0].age));
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

ImprovedIndividual ImprovedGeneticTrainer::getBestIndividual() const {
    if (population.empty()) {
        throw std::runtime_error("Популяция пуста");
    }
    
    return population[0];
}

double ImprovedGeneticTrainer::getBestLoss() const {
    if (population.empty()) {
        return 0.0;
    }
    
    return population[0].loss;
}

double ImprovedGeneticTrainer::getAverageFitness() const {
    if (population.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (const auto& ind : population) {
        sum += ind.fitness;
    }
    
    return sum / population.size();
}

void ImprovedGeneticTrainer::setParameters(size_t popSize, size_t maxGen, double mutRate,
                                          double crossRate, double eliteRatio, double lr) {
    populationSize = popSize;
    maxGenerations = maxGen;
    mutationRate = mutRate;
    crossoverRate = crossRate;
    elitismRatio = eliteRatio;
    learningRate = lr;
}