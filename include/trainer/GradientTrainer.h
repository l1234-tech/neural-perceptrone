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

class GradientTrainer : public ITrainer {
private:
    // Параметры обучения
    double learningRate;          // Скорость обучения
    size_t epochs;               // Количество эпох
    size_t batchSize;            // Размер мини-батча (0 = полный батч)
    double lambda;               // Коэффициент L2 регуляризации
    size_t earlyStoppingPatience; // Терпение ранней остановки
    double validationSplit;      // Доля данных для валидации (если validationData пусто)
    bool shuffle;                // Перемешивать данные каждую эпоху
    
    // Состояние обучения
    std::vector<double> trainLossHistory;
    std::vector<double> valLossHistory;
    std::vector<double> valF1History;
    DataSet validationData;      // Валидационная выборка (если предоставлена)
    
    // Вспомогательные методы
    double computeBatchLoss(NeuralNetwork& network, const DataSet& batch);
    void updateWeightsWithRegularization(NeuralNetwork& network, const std::vector<std::vector<double>>& gradients, double learningRate);
    
public:
    // Конструктор с параметрами по умолчанию
    GradientTrainer(double learningRate = 0.01,
                    size_t epochs = 100,
                    size_t batchSize = 32,
                    double lambda = 0.01,
                    size_t earlyStoppingPatience = 10,
                    double validationSplit = 0.2,
                    bool shuffle = true);
    
    // Основной метод обучения
    std::vector<double> train(NeuralNetwork& network, 
                             const DataSet& trainingData,
                             const DataSet& validationData = DataSet()) override;
    
    // Дообучение на новых данных с уменьшенной скоростью обучения
    std::vector<double> continueTraining(NeuralNetwork& network, 
                                         const DataSet& trainingData,
                                         const DataSet& validationData = DataSet()) override;
    
    // Получить конфигурацию в виде строки
    std::string getConfig() const override;
    
    // Сохранение и загрузка состояния (не реализовано для градиентного спуска)
    void saveState(const std::string& filename) const override;
    void loadState(const std::string& filename) override;
    
    // Получить метрики обучения
    std::vector<std::pair<std::string, double>> getMetrics() const override;
    
    // Геттеры для истории ошибок
    const std::vector<double>& getTrainLossHistory() const { return trainLossHistory; }
    const std::vector<double>& getValLossHistory() const { return valLossHistory; }
    const std::vector<double>& getValF1History() const { return valF1History; }
    
    // Сеттеры параметров
    void setLearningRate(double rate) { learningRate = rate; }
    void setEpochs(size_t epochs) { this->epochs = epochs; }
    void setBatchSize(size_t batchSize) { this->batchSize = batchSize; }
    void setLambda(double lambda) { this->lambda = lambda; }
    void setEarlyStoppingPatience(size_t patience) { earlyStoppingPatience = patience; }
    void setValidationSplit(double split) { validationSplit = split; }
    void setShuffle(bool shuffle) { this->shuffle = shuffle; }
    void setValidationData(const DataSet& data) { validationData = data; }
    
    // Оценка метрик на выборке
    ClassificationMetrics evaluateMetrics(NeuralNetwork& network, const DataSet& data) const;
};