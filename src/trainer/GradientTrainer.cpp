#include "trainer/GradientTrainer.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataLoader.h"
#include "metrics/ClassificationMetrics.h"
#include "utils/Random.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <limits>

GradientTrainer::GradientTrainer(double learningRate,
                                 size_t epochs,
                                 size_t batchSize,
                                 double lambda,
                                 size_t earlyStoppingPatience,
                                 double validationSplit,
                                 bool shuffle)
    : learningRate(learningRate)
    , epochs(epochs)
    , batchSize(batchSize)
    , lambda(lambda)
    , earlyStoppingPatience(earlyStoppingPatience)
    , validationSplit(validationSplit)
    , shuffle(shuffle) {
    
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Скорость обучения должна быть положительной");
    }
    if (validationSplit < 0.0 || validationSplit >= 1.0) {
        throw std::invalid_argument("Доля валидации должна быть в диапазоне [0, 1)");
    }
}

std::vector<double> GradientTrainer::train(NeuralNetwork& network, 
                                          const DataSet& trainingData,
                                          const DataSet& validationData) {
    // Очищаем историю
    trainLossHistory.clear();
    valLossHistory.clear();
    valF1History.clear();
    
    // Подготавливаем валидационную выборку
    DataSet valData = validationData;
    DataSet trainData = trainingData;
    
    // Если валидационная выборка не предоставлена, разделяем обучающую
    if (valData.empty() && validationSplit > 0.0) {
        auto [trainSplit, valSplit] = trainingData.split(1.0 - validationSplit);
        trainData = trainSplit;
        valData = valSplit;
        std::cout << "Данные разделены: " << trainData.size() 
                  << " примеров для обучения, " << valData.size() 
                  << " примеров для валидации\n";
    }
    
    // Устанавливаем валидационную выборку
    this->validationData = valData;
    
    std::cout << "Начало обучения градиентным спуском...\n";
    std::cout << "Параметры: learningRate=" << learningRate 
              << ", epochs=" << epochs 
              << ", batchSize=" << batchSize 
              << ", lambda=" << lambda 
              << ", earlyStoppingPatience=" << earlyStoppingPatience << "\n";
    
    size_t bestEpoch = 0;
    double bestValLoss = std::numeric_limits<double>::max();
    size_t patienceCounter = 0;
    
    auto startTime = std::chrono::steady_clock::now();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Перемешиваем данные в начале каждой эпохи, если включено
        if (shuffle) {
            trainData.shuffle();
        }
        
        double epochLoss = 0.0;
        size_t numBatches = 0;
        
        // Обработка мини-батчей
        if (batchSize == 0 || batchSize >= trainData.size()) {
            // Полный батч
            // Вычисляем градиенты для всего набора данных
            // Пока используем простой подход: обновляем веса для каждого примера отдельно
            // (можно оптимизировать позже)
            for (size_t i = 0; i < trainData.size(); ++i) {
                const auto& point = trainData.get(i);
                const auto& features = point.getFeatures();
                
                Vector input(features.size());
                for (size_t j = 0; j < features.size(); ++j) {
                    input[j] = features[j];
                }
                
                Vector target(network.getArchitecture().back(), 0.0);
                int label = point.getLabel();
                if (network.getArchitecture().back() == 1) {
                    target[0] = static_cast<double>(label);
                } else {
                    if (label >= 0 && static_cast<size_t>(label) < target.size()) {
                        target[label] = 1.0;
                    }
                }
                
                double loss = network.trainOneExample(input, target, learningRate);
                epochLoss += loss;
            }
            numBatches = 1;
        } else {
            // Мини-батчи
            for (size_t start = 0; start < trainData.size(); start += batchSize) {
                size_t end = std::min(start + batchSize, trainData.size());
                
                // Для каждого примера в батче выполняем обновление
                // (упрощенная реализация - можно оптимизировать для вычисления среднего градиента)
                for (size_t i = start; i < end; ++i) {
                    const auto& point = trainData.get(i);
                    const auto& features = point.getFeatures();
                    
                    Vector input(features.size());
                    for (size_t j = 0; j < features.size(); ++j) {
                        input[j] = features[j];
                    }
                    
                    Vector target(network.getArchitecture().back(), 0.0);
                    int label = point.getLabel();
                    if (network.getArchitecture().back() == 1) {
                        target[0] = static_cast<double>(label);
                    } else {
                        if (label >= 0 && static_cast<size_t>(label) < target.size()) {
                            target[label] = 1.0;
                        }
                    }
                    
                    double loss = network.trainOneExample(input, target, learningRate);
                    epochLoss += loss;
                }
                numBatches++;
            }
        }
        
        // Средняя ошибка на эпохе
        double avgEpochLoss = epochLoss / trainData.size();
        trainLossHistory.push_back(avgEpochLoss);
        
        // Вычисляем ошибку на валидационной выборке
        double valLoss = std::numeric_limits<double>::max();
        double valF1 = 0.0;
        if (!valData.empty()) {
            valLoss = network.computeLoss(valData, lambda);
            valLossHistory.push_back(valLoss);
            
            // Вычисляем F1-score на валидации
            ClassificationMetrics metrics = evaluateMetrics(network, valData);
            valF1 = metrics.macroF1;
            valF1History.push_back(valF1);
        }
        
        // Логирование каждые 10 эпох или на первой/последней эпохе
        if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0) {
            std::cout << "Эпоха " << (epoch + 1) << "/" << epochs 
                      << ": train loss = " << avgEpochLoss;
            if (!valData.empty()) {
                std::cout << ", val loss = " << valLoss << ", val F1 = " << valF1;
            }
            std::cout << "\n";
        }
        
        // Ранняя остановка по валидационной ошибке
        if (!valData.empty()) {
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (earlyStoppingPatience > 0 && patienceCounter >= earlyStoppingPatience) {
                    std::cout << "Ранняя остановка на эпохе " << (epoch + 1) 
                              << " (лучшая эпоха " << (bestEpoch + 1) 
                              << " с val loss = " << bestValLoss << ")\n";
                    break;
                }
            }
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Обучение завершено за " << duration.count() / 1000.0 << " секунд\n";
    
    // Возвращаем историю ошибок на обучении
    return trainLossHistory;
}

std::string GradientTrainer::getConfig() const {
    std::string config = "Gradient Trainer Configuration:\n";
    config += "  Learning rate: " + std::to_string(learningRate) + "\n";
    config += "  Epochs: " + std::to_string(epochs) + "\n";
    config += "  Batch size: " + std::to_string(batchSize) + "\n";
    config += "  L2 regularization (lambda): " + std::to_string(lambda) + "\n";
    config += "  Early stopping patience: " + std::to_string(earlyStoppingPatience) + "\n";
    config += "  Validation split: " + std::to_string(validationSplit) + "\n";
    config += "  Shuffle: " + std::string(shuffle ? "true" : "false") + "\n";
    return config;
}

void GradientTrainer::saveState(const std::string& filename) const {
    // В будущей реализации можно сохранять веса и состояние оптимизатора
    std::cout << "Сохранение состояния градиентного тренера не реализовано\n";
    // Пока просто создаем пустой файл, чтобы избежать ошибок
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "GradientTrainer state\n";
        file.close();
    }
}

void GradientTrainer::loadState(const std::string& filename) {
    // В будущей реализации можно загружать веса и состояние оптимизатора
    std::cout << "Загрузка состояния градиентного тренера не реализовано\n";
}

std::vector<std::pair<std::string, double>> GradientTrainer::getMetrics() const {
    std::vector<std::pair<std::string, double>> metrics;
    
    if (!trainLossHistory.empty()) {
        metrics.emplace_back("final_train_loss", trainLossHistory.back());
    }
    
    if (!valLossHistory.empty()) {
        metrics.emplace_back("final_val_loss", valLossHistory.back());
    }
    
    if (!valF1History.empty()) {
        metrics.emplace_back("final_val_f1", valF1History.back());
    }
    
    // Добавляем минимальные и средние значения
    if (!trainLossHistory.empty()) {
        double minTrainLoss = *std::min_element(trainLossHistory.begin(), trainLossHistory.end());
        double avgTrainLoss = std::accumulate(trainLossHistory.begin(), trainLossHistory.end(), 0.0) / trainLossHistory.size();
        metrics.emplace_back("min_train_loss", minTrainLoss);
        metrics.emplace_back("avg_train_loss", avgTrainLoss);
    }
    
    if (!valLossHistory.empty()) {
        double minValLoss = *std::min_element(valLossHistory.begin(), valLossHistory.end());
        double avgValLoss = std::accumulate(valLossHistory.begin(), valLossHistory.end(), 0.0) / valLossHistory.size();
        metrics.emplace_back("min_val_loss", minValLoss);
        metrics.emplace_back("avg_val_loss", avgValLoss);
    }
    
    return metrics;
}

ClassificationMetrics GradientTrainer::evaluateMetrics(NeuralNetwork& network, const DataSet& data) const {
    return calculateMetrics(network, data);
}