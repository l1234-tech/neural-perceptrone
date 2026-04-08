#pragma once
#include <vector>
#include <memory>
#include "data/DataSet.h"
#include <string>

// Предварительное объявление классов
class NeuralNetwork;
class DataSet;

// Интерфейс тренера (абстрактный класс)
// Определяет общий контракт для всех алгоритмов обучения нейронных сетей
class ITrainer {
public:
    virtual ~ITrainer() = default;

    // Обучить нейронную сеть на датасете
    // Возвращает историю ошибок (loss) на каждой эпохе
    virtual std::vector<double> train(NeuralNetwork& network, 
                                      const DataSet& trainingData,
                                      const DataSet& validationData = DataSet()) = 0;

    // Получить текущую конфигурацию обучения
    virtual std::string getConfig() const = 0;

    // Сохранить состояние тренера в файл
    virtual void saveState(const std::string& filename) const = 0;

    // Загрузить состояние тренера из файла
    virtual void loadState(const std::string& filename) = 0;

    // Получить метрики обучения (точность, ошибка и т.д.)
    virtual std::vector<std::pair<std::string, double>> getMetrics() const = 0;
};