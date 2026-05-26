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
        throw std::invalid_argument("Population size must be at least 10");
    }
    if (elitismRatio < 0.0 || elitismRatio > 0.5) {
        throw std::invalid_argument("Elitism ratio must be in range [0, 0.5]");
    }
    if (regularizationLambda < 0.0) {
        throw std::invalid_argument("Regularization coefficient must be non-negative");
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
        
        // Calculate loss on training data with L2 regularization
        individual.loss = network.computeLoss(trainingData, regularizationLambda);
        
        // SIMPLE fitness function: use F1-score on validation OR inverse loss
        if (useF1Fitness && !validationData.empty()) {
            // Use F1-score on validation set as fitness
            ClassificationMetrics metrics = calculateMetrics(network, validationData);
            individual.f1Score = metrics.macroF1;
            individual.fitness = individual.f1Score + 1e-8; // Add epsilon to avoid zero
        } else {
            // Use inverse loss as fitness (traditional approach)
            individual.f1Score = 0.0;
            individual.fitness = 1.0 / (individual.loss + 1e-8);
        }
    }
    
    // Restore original weights
    network.setWeights(originalWeights);
    
    // Sort population by fitness
    std::sort(population.begin(), population.end());
}

void GeneticTrainer::tournamentSelection(size_t tournamentSize) {
    std::vector<GeneticIndividual> newPopulation;
    newPopulation.reserve(populationSize);
    
    // Preserve elite individuals
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
                
                // Limit weight values
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
        throw std::invalid_argument("Training dataset cannot be empty");
    }
    
    lossHistory.clear();
    bestLossHistory.clear();
    validationLossHistory.clear();
    validationF1History.clear();
    
    // Save validation dataset for F1 calculation
    this->validationData = validationData;
    
    initializePopulation(network);
    
    size_t noImprovementCount = 0;
    double bestValidationF1 = 0.0;
    double bestValidationLoss = std::numeric_limits<double>::max();
    
    for (size_t generation = 0; generation < maxGenerations; ++generation) {
        // Fitness evaluation
        evaluatePopulation(network, trainingData);
        
        // Save statistics
        double bestLoss = population[0].loss;
        double avgLoss = 0.0;
        
        for (const auto& ind : population) {
            avgLoss += ind.loss;
        }
        avgLoss /= population.size();
        
        lossHistory.push_back(avgLoss);
        bestLossHistory.push_back(bestLoss);
        
        // Progress output every 20 generations
        if (generation % 20 == 0) {
            std::cout << "Generation " << std::setw(3) << generation 
                      << ": best loss = " << std::fixed << std::setprecision(6) << bestLoss
                      << ", average loss = " << avgLoss 
                      << ", best fitness = " << std::setprecision(4) << population[0].fitness;
            
            if (useF1Fitness && !validationData.empty() && population[0].f1Score > 0) {
                std::cout << ", F1 = " << std::setprecision(4) << population[0].f1Score;
            }
            std::cout << std::endl;
        }
        
        // Early stopping on validation (if validation data exists)
        if (!validationData.empty()) {
            std::vector<double> originalWeights = network.getWeights();
            network.setWeights(population[0].chromosome);
            
            // Calculate loss on validation
            double validationLoss = network.computeLoss(validationData, 0.0); // without regularization for evaluation
            validationLossHistory.push_back(validationLoss);
            
            // Calculate F1-score on validation
            ClassificationMetrics metrics = calculateMetrics(network, validationData);
            double validationF1 = metrics.macroF1;
            validationF1History.push_back(validationF1);
            
            network.setWeights(originalWeights);
            
            // Early stopping on F1 (if using F1 as fitness)
            if (useF1Fitness) {
                if (validationF1 > bestValidationF1) {
                    bestValidationF1 = validationF1;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            } else {
                // Early stopping on loss
                if (validationLoss < bestValidationLoss) {
                    bestValidationLoss = validationLoss;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            }
            
            if (noImprovementCount >= earlyStoppingPatience) {
                std::cout << "Early stopping: no improvement on validation for " 
                          << earlyStoppingPatience << " generations" << std::endl;
                break;
            }
        }
        
        tournamentSelection();
        uniformCrossover();
        simpleMutation();
    }
    
    // Set best weights to network
    network.setWeights(population[0].chromosome);
    
    // Evaluate on validation set if provided
    if (!validationData.empty() && validationLossHistory.empty()) {
        double validationLoss = network.computeLoss(validationData, 0.0);
        std::cout << "Loss on validation dataset: " << validationLoss << std::endl;
    }
    
    return bestLossHistory;
}

std::vector<double> GeneticTrainer::continueTraining(NeuralNetwork& network,
                                                    const DataSet& trainingData,
                                                    const DataSet& validationData) {
    std::cout << "Fine-tuning model with genetic algorithm on new data...\n";
    
    // Save current histories and parameters
    std::vector<double> oldLossHistory = lossHistory;
    std::vector<double> oldBestLossHistory = bestLossHistory;
    std::vector<double> oldValidationLossHistory = validationLossHistory;
    std::vector<double> oldValidationF1History = validationF1History;
    double originalMutationRate = mutationRate;
    size_t originalMaxGenerations = maxGenerations;
    
    // Reduce mutation probability for fine-tuning (weights are already good)
    double fineTuneMutationRate = originalMutationRate * 0.3;
    if (fineTuneMutationRate < 0.01) fineTuneMutationRate = 0.01;
    mutationRate = fineTuneMutationRate;
    
    // Reduce number of generations for fine-tuning
    size_t fineTuneGenerations = originalMaxGenerations / 2;
    if (fineTuneGenerations < 10) fineTuneGenerations = 10;
    maxGenerations = fineTuneGenerations;
    
    std::cout << "Fine-tuning parameters: mutationRate reduced from " << originalMutationRate
              << " to " << fineTuneMutationRate
              << ", maxGenerations reduced from " << originalMaxGenerations
              << " to " << fineTuneGenerations << "\n";
    
    // Train on new data (history will be cleared in train method)
    std::vector<double> newHistory = train(network, trainingData, validationData);
    
    // Restore original parameters
    mutationRate = originalMutationRate;
    maxGenerations = originalMaxGenerations;
    
    // Merge histories
    lossHistory.insert(lossHistory.begin(), oldLossHistory.begin(), oldLossHistory.end());
    bestLossHistory.insert(bestLossHistory.begin(), oldBestLossHistory.begin(), oldBestLossHistory.end());
    validationLossHistory.insert(validationLossHistory.begin(), oldValidationLossHistory.begin(), oldValidationLossHistory.end());
    validationF1History.insert(validationF1History.begin(), oldValidationF1History.begin(), oldValidationF1History.end());
    
    // Merge returned history
    std::vector<double> combinedHistory = oldBestLossHistory;
    combinedHistory.insert(combinedHistory.end(), newHistory.begin(), newHistory.end());
    
    std::cout << "Fine-tuning completed. Total training history now contains " 
              << combinedHistory.size() << " generations.\n";
    
    return combinedHistory;
}

std::string GeneticTrainer::getConfig() const {
    std::ostringstream ss;
    ss << "Genetic Algorithm (simplified):\n"
       << "  Population size: " << populationSize << "\n"
       << "  Maximum generations: " << maxGenerations << "\n"
       << "  Mutation rate: " << mutationRate << "\n"
       << "  Crossover rate: " << crossoverRate << "\n"
       << "  Elitism ratio: " << elitismRatio << "\n"
       << "  L2 regularization coefficient: " << regularizationLambda << "\n"
       << "  Early stopping patience: " << earlyStoppingPatience << "\n"
       << "  Use F1 as fitness: " << (useF1Fitness ? "yes" : "no") << "\n"
       << "  Current population size: " << population.size() << "\n"
       << "  Loss history: " << lossHistory.size() << " records";
    
    return ss.str();
}

void GeneticTrainer::saveState(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving: " + filename);
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
        throw std::runtime_error("Failed to open file for loading: " + filename);
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
        metrics.emplace_back("Best fitness", population[0].fitness);
        metrics.emplace_back("Best loss", population[0].loss);
        
        if (useF1Fitness && !validationData.empty()) {
            metrics.emplace_back("Best F1-score", population[0].f1Score);
        }
        
        double avgFitness = 0.0;
        for (const auto& ind : population) {
            avgFitness += ind.fitness;
        }
        avgFitness /= population.size();
        metrics.emplace_back("Average fitness", avgFitness);
    }
    
    if (!lossHistory.empty()) {
        metrics.emplace_back("Initial loss", lossHistory[0]);
        metrics.emplace_back("Final loss", lossHistory.back());
        
        if (lossHistory.size() > 1) {
            double improvement = (lossHistory[0] - lossHistory.back()) / lossHistory[0] * 100.0;
            metrics.emplace_back("Loss improvement (%)", improvement);
        }
    }
    
    if (!validationLossHistory.empty()) {
        metrics.emplace_back("Best validation loss", *std::min_element(validationLossHistory.begin(), validationLossHistory.end()));
        metrics.emplace_back("Final validation loss", validationLossHistory.back());
    }
    
    if (!validationF1History.empty()) {
        metrics.emplace_back("Best validation F1", *std::max_element(validationF1History.begin(), validationF1History.end()));
        metrics.emplace_back("Final validation F1", validationF1History.back());
    }
    
    metrics.emplace_back("Population size", static_cast<double>(population.size()));
    metrics.emplace_back("Generations completed", static_cast<double>(lossHistory.size()));
    
    return metrics;
}

GeneticIndividual GeneticTrainer::getBestIndividual() const {
    if (population.empty()) {
        throw std::runtime_error("Population is empty");
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
        throw std::runtime_error("Population is empty");
    }
    
    // Save current weights
    std::vector<double> originalWeights = network.getWeights();
    
    // Set best weights
    network.setWeights(population[0].chromosome);
    
    // Calculate metrics
    ClassificationMetrics metrics = calculateMetrics(network, data);
    
    // Restore original weights
    network.setWeights(originalWeights);
    
    return metrics;
}