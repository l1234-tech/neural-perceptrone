#include "neural_network/NeuralNetwork.h"
#include "trainer/GradientTrainer.h"
#include "trainer/GeneticTrainer.h"
#include "data/DataLoader.h"
#include "data/DataSet.h"
#include "metrics/ClassificationMetrics.h"
#include "utils/Random.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>

enum TrainingAlgorithm { GRADIENT, GENETIC };

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

NeuralNetwork createNetwork(size_t featureCount, const std::string& datasetHint = "") {
    // ПРОСТЫЕ архитектуры для предотвращения переобучения
    std::vector<size_t> architecture;
    
    if (featureCount == 2) {
        // Для всех датасетов с 2 признаками (train.csv, d1.csv) - ОЧЕНЬ простая архитектура
        // Архитектура: [2, 2, 1] - всего 7 параметров
        architecture = {2, 2, 1};
        std::cout << "Используется простая архитектура [2, 2, 1] (7 параметров)\n";
    } else if (featureCount == 4) {
        // Для d2.csv (4 признака) - простая архитектура
        // Архитектура: [4, 3, 1] - всего 19 параметров
        architecture = {4, 3, 1};
        std::cout << "Используется простая архитектура [4, 3, 1] (19 параметров)\n";
    } else {
        // По умолчанию: [featureCount, 2, 1]
        architecture = {featureCount, 2, 1};
    }
    
    NeuralNetwork network(architecture);
    
    std::cout << "Архитектура сети: [";
    for (size_t i = 0; i < architecture.size(); ++i) {
        std::cout << architecture[i];
        if (i < architecture.size() - 1) std::cout << " → ";
    }
    std::cout << "]\n";
    std::cout << "Всего параметров: " << network.getTotalParameters() << "\n";
    
    return network;
}

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

DataSet loadDataset(const std::string& filename, bool hasLabels = true) {
    std::cout << "\nЗагрузка данных из: " << filename << "\n";
    
    if (!fileExists(filename)) {
        throw std::runtime_error("Файл не найден: " + filename);
    }
    
    // Проверяем первые несколько строк файла для диагностики
    std::ifstream testFile(filename);
    std::string line;
    int lineCount = 0;
    while (std::getline(testFile, line) && lineCount < 5) {
        std::cout << "Строка " << lineCount << ": " << line << "\n";
        lineCount++;
    }
    testFile.close();
    
    DataSet dataset;
    if (hasLabels) {
        dataset = DataLoader::loadFromCSV(filename);
    } else {
        dataset = DataLoader::loadFromCSVNoLabels(filename);
    }
    
    size_t featureCount = dataset.getFeatureCount();
    
    std::cout << "Загружено примеров: " << dataset.size() << "\n";
    std::cout << "Количество признаков: " << featureCount << "\n";
    
    if (dataset.size() == 0) {
        throw std::runtime_error("Датасет пуст после загрузки. Возможно, некорректный формат CSV.");
    }
    
    return dataset;
}

NeuralNetwork trainModel(NeuralNetwork& network, const DataSet& trainData, const std::string& datasetName, TrainingAlgorithm algorithm) {
    printHeader("ОБУЧЕНИЕ МОДЕЛИ" + (datasetName.empty() ? "" : " (" + datasetName + ")"));
    
    if (algorithm == GRADIENT) {
        // Параметры градиентного спуска
        GradientTrainer trainer;
        
        if (datasetName.find("train") != std::string::npos) {
            // Оптимизированные параметры для train.csv (маленький датасет)
            trainer = GradientTrainer(
                0.1,    // learning rate (высокий для быстрой сходимости)
                200,    // epochs (больше эпох для маленького датасет)
                8,      // batch size (маленький батч)
                0.05,   // lambda (L2 регуляризация)
                0,      // early stopping patience (отключено, т.к. нет валидации)
                0.0,    // validation split (0 = нет валидации)
                true    // shuffle
            );
            std::cout << "\nИспользуются консервативные параметры для train.csv (градиентный спуск)\n";
        } else {
            // Стандартные параметры для d1.csv и d2.csv
            trainer = GradientTrainer(
                0.05,   // learning rate
                150,    // epochs
                16,     // batch size
                0.1,    // lambda (L2 регуляризация)
                0,      // early stopping patience (отключено)
                0.0,    // validation split (0 = нет валидации)
                true    // shuffle
            );
            std::cout << "\nИспользуются стандартные параметры для датасетов d1/d2 (градиентный спуск)\n";
        }
        
        std::cout << "\nКонфигурация алгоритма:\n" << trainer.getConfig() << "\n";
        
        std::cout << "\nНачало обучения...\n";
        auto startTime = std::chrono::steady_clock::now();
        
        // Обучаем без валидационной выборки
        auto lossHistory = trainer.train(network, trainData, DataSet());
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "\nОбучение завершено за " << duration.count() / 1000.0 << " секунд\n";
        
        if (!lossHistory.empty()) {
            double initialLoss = lossHistory[0];
            double finalLoss = lossHistory.back();
            double improvement = (initialLoss - finalLoss) / initialLoss * 100.0;
            
            std::cout << "Начальная ошибка: " << std::fixed << std::setprecision(6) << initialLoss << "\n";
            std::cout << "Финальная ошибка: " << finalLoss << "\n";
            std::cout << "Улучшение: " << std::setprecision(1) << improvement << "%\n";
            
            if (lossHistory.size() > 5) {
                std::cout << "\nПрогресс обучения:\n";
                for (size_t i = 0; i < lossHistory.size(); i += lossHistory.size() / 5) {
                    std::cout << "  Эпоха " << std::setw(3) << i 
                              << ": ошибка = " << std::setprecision(6) << lossHistory[i] << "\n";
                }
            }
        }
    } else { // GENETIC
        // Параметры генетического алгоритма
        GeneticTrainer trainer;
        
        if (datasetName.find("train") != std::string::npos) {
            // Оптимизированные параметры для train.csv (маленький датасет)
            trainer = GeneticTrainer(
                30,     // populationSize (меньше для маленького датасета)
                50,     // maxGenerations (меньше поколений)
                0.15,   // mutationRate (выше мутация)
                0.7,    // crossoverRate
                0.1,    // elitismRatio
                0.05,   // regularizationLambda
                5,      // earlyStoppingPatience
                true    // useF1Fitness
            );
            std::cout << "\nИспользуются консервативные параметры для train.csv (генетический алгоритм)\n";
        } else {
            // Стандартные параметры для d1.csv и d2.csv
            trainer = GeneticTrainer(
                50,     // populationSize
                100,    // maxGenerations
                0.1,    // mutationRate
                0.8,    // crossoverRate
                0.1,    // elitismRatio
                0.01,   // regularizationLambda
                10,     // earlyStoppingPatience
                true    // useF1Fitness
            );
            std::cout << "\nИспользуются стандартные параметры для датасетов d1/d2 (генетический алгоритм)\n";
        }
        
        std::cout << "\nКонфигурация алгоритма:\n" << trainer.getConfig() << "\n";
        
        std::cout << "\nНачало обучения...\n";
        auto startTime = std::chrono::steady_clock::now();
        
        // Обучаем без валидационной выборки
        auto lossHistory = trainer.train(network, trainData, DataSet());
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "\nОбучение завершено за " << duration.count() / 1000.0 << " секунд\n";
        
        if (!lossHistory.empty()) {
            double initialLoss = lossHistory[0];
            double finalLoss = lossHistory.back();
            double improvement = (initialLoss - finalLoss) / initialLoss * 100.0;
            
            std::cout << "Начальная ошибка: " << std::fixed << std::setprecision(6) << initialLoss << "\n";
            std::cout << "Финальная ошибка: " << finalLoss << "\n";
            std::cout << "Улучшение: " << std::setprecision(1) << improvement << "%\n";
            
            if (lossHistory.size() > 5) {
                std::cout << "\nПрогресс обучения:\n";
                for (size_t i = 0; i < lossHistory.size(); i += lossHistory.size() / 5) {
                    std::cout << "  Эпоха " << std::setw(3) << i 
                              << ": ошибка = " << std::setprecision(6) << lossHistory[i] << "\n";
                }
            }
        }
        
        // Дополнительная информация для генетического алгоритма
        double bestLoss = trainer.getBestLoss();
        double bestF1 = trainer.getBestF1Score();
        std::cout << "Лучшая ошибка в популяции: " << bestLoss << "\n";
        std::cout << "Лучший F1-score в популяции: " << bestF1 << "\n";
    }
    
    return network;
}

ClassificationMetrics evaluateModel(NeuralNetwork& network, const DataSet& testData, const std::string& datasetName) {
    printHeader("ОЦЕНКА МОДЕЛИ: " + datasetName);
    
    ClassificationMetrics metrics = calculateMetrics(network, testData);
    printMetrics(metrics);
    
    return metrics;
}

int main() {
    try {
        initRandom(42);  // seed 
        
        system("chcp 65001 > nul");
        
        printHeader("ЛАБОРАТОРНАЯ РАБОТА: КЛАССИФИКАЦИЯ");
        
        // Запрос выбора алгоритм�� обучения
        std::cout << "\nВыберите алгоритм обучения:\n";
        std::cout << "1 - Градиентный спуск\n";
        std::cout << "2 - Генетический алгоритм\n";
        std::cout << "Введите номер (1 или 2): ";
        
        int choice;
        std::cin >> choice;
        if (choice != 1 && choice != 2) {
            std::cout << "Неверный выбор. По умолчанию используется градиентный спуск.\n";
            choice = 1;
        }
        
        TrainingAlgorithm algorithm = (choice == 1) ? GRADIENT : GENETIC;
        std::cout << "\nВыбран алгоритм: " << (algorithm == GRADIENT ? "Градиентный спуск" : "Генетический алгоритм") << "\n";
        
        // Загружаем полные датасеты и делим на обучающую и тестовую части (80/20)
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (80/20 split)\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Датасет train.csv (2 признака)
        std::cout << "\nЗагрузка train.csv...\n";
        DataSet trainFullDataset = loadDataset("data/train.csv", true);
        auto [trainTrainData, trainTestData] = trainFullDataset.split(0.8);
        std::cout << "  Обучающая часть (80%): " << trainTrainData.size() << " примеров\n";
        std::cout << "  Тестовая часть (20%): " << trainTestData.size() << " примеров\n";
        
        // Датасет d1.csv (2 признака)
        std::cout << "\nЗагрузка d1.csv...\n";
        DataSet d1FullDataset = loadDataset("data/d1.csv", true);
        auto [d1TrainData, d1TestData] = d1FullDataset.split(0.8);
        std::cout << "  Обучающая часть (80%): " << d1TrainData.size() << " примеров\n";
        std::cout << "  Тестовая часть (20%): " << d1TestData.size() << " примеров\n";
        
        // Датасет d2.csv (4 признака)
        std::cout << "\nЗагрузка d2.csv...\n";
        DataSet d2FullDataset = loadDataset("data/d2.csv", true);
        auto [d2TrainData, d2TestData] = d2FullDataset.split(0.8);
        std::cout << "  Обучающая часть (80%): " << d2TrainData.size() << " примеров\n";
        std::cout << "  Тестовая часть (20%): " << d2TestData.size() << " примеров\n";
        
        // Датасет test.csv (2 признака, без меток) - загружаем отдельно для предсказаний
        std::cout << "\nЗагрузка test.csv (без меток)...\n";
        DataSet testDataset = loadDataset("data/test.csv", false);
        std::cout << "  Загружено примеров: " << testDataset.size() << "\n";
        
        // ОБУЧЕНИЕ МОДЕЛЕЙ
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ОБУЧЕНИЕ МОДЕЛЕЙ\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Модель для train.csv
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ train.csv (2 признака) - ОБУЧЕНИЕ\n";
        std::cout << std::string(60, '-') << "\n";
        NeuralNetwork network1 = createNetwork(trainTrainData.getFeatureCount(), "train");
        trainModel(network1, trainTrainData, "train.csv", algorithm);
        
        // Модель для d1.csv
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ d1.csv (2 признака) - ОБУЧЕНИЕ\n";
        std::cout << std::string(60, '-') << "\n";
        NeuralNetwork network2 = createNetwork(d1TrainData.getFeatureCount(), "d1");
        trainModel(network2, d1TrainData, "d1.csv", algorithm);
        
        // Модель для d2.csv
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ d2.csv (4 признака) - ОБУЧЕНИЕ\n";
        std::cout << std::string(60, '-') << "\n";
        NeuralNetwork network3 = createNetwork(d2TrainData.getFeatureCount(), "d2");
        trainModel(network3, d2TrainData, "d2.csv", algorithm);
        
        // ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ДАННЫХ (20%) И test.csv
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ДАННЫХ (20%) И test.csv\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Предсказания на test.csv (2 признака, без меток)
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ test.csv (2 признака, предсказания)\n";
        std::cout << std::string(60, '-') << "\n";
        
        // Используем модель, обученную на train.csv, для предсказаний на test.csv
        std::cout << "\nИспользуем модель, обученную на train.csv, для предсказаний на test.csv\n";
        
        // Проверяем совпадение количества признаков
        if (network1.getArchitecture()[0] == testDataset.getFeatureCount()) {
            std::cout << "\nПредсказания для test.csv (первые 15 примеров):\n";
            std::cout << "x1	x2	Предсказание (0/1)\n";
            for (size_t i = 0; i < std::min(size_t(15), testDataset.size()); ++i) {
                const auto& point = testDataset.get(i);
                const auto& features = point.getFeatures();
                
                // Создаем Vector из features
                Vector input(features.size());
                for (size_t j = 0; j < features.size(); ++j) {
                    input[j] = features[j];
                }
                
                int prediction = network1.predict(input);
                std::cout << std::fixed << std::setprecision(4) 
                          << features[0] << "	" << features[1] << "	" 
                          << prediction << "\n";
            }
            std::cout << "\nВсего сделано предсказаний: " << testDataset.size() << "\n";
        } else {
            std::cout << "Ошибка: количество признаков в test.csv (" << testDataset.getFeatureCount() 
                      << ") не совпадает с входным размером модели (" << network1.getArchitecture()[0] << ")\n";
        }
        
        // Предсказания на d1_test (20% без меток) - используем тестовую часть d1
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ d1_test (20% без меток, 2 признака)\n";
        std::cout << std::string(60, '-') << "\n";
        
        if (network2.getArchitecture()[0] == d1TestData.getFeatureCount()) {
            std::cout << "\nПредсказания для d1_test (первые 10 примеров):\n";
            std::cout << "feature_0	feature_1	Предсказание (0/1)\n";
            for (size_t i = 0; i < std::min(size_t(10), d1TestData.size()); ++i) {
                const auto& point = d1TestData.get(i);
                const auto& features = point.getFeatures();
                
                Vector input(features.size());
                for (size_t j = 0; j < features.size(); ++j) {
                    input[j] = features[j];
                }
                
                int prediction = network2.predict(input);
                std::cout << std::fixed << std::setprecision(4) 
                          << features[0] << "	" << features[1] << "	" 
                          << prediction << "\n";
            }
            std::cout << "Всего предсказаний для d1_test: " << d1TestData.size() << "\n";
        }
        
        // Предсказания на d2_test (20% без меток) - используем тестовую часть d2
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ d2_test (20% без меток, 4 признака)\n";
        std::cout << std::string(60, '-') << "\n";
        
        if (network3.getArchitecture()[0] == d2TestData.getFeatureCount()) {
            std::cout << "\nПредсказания для d2_test (первые 10 примеров):\n";
            std::cout << "feature_0	feature_1	feature_2	feature_3	Предсказание (0/1)\n";
            for (size_t i = 0; i < std::min(size_t(10), d2TestData.size()); ++i) {
                const auto& point = d2TestData.get(i);
                const auto& features = point.getFeatures();
                
                Vector input(features.size());
                for (size_t j = 0; j < features.size(); ++j) {
                    input[j] = features[j];
                }
                
                int prediction = network3.predict(input);
                std::cout << std::fixed << std::setprecision(4) 
                          << features[0] << "	" << features[1] << "	" 
                          << features[2] << "	" << features[3] << "	"
                          << prediction << "\n";
            }
            std::cout << "Всего предсказаний для d2_test: " << d2TestData.size() << "\n";
        }
        
        // Итоговая оценка (на основе тестовых данных d1 и d2 с метками)
        printHeader("ИТОГОВАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ (20%)");
        
        ClassificationMetrics testMetrics2 = evaluateModel(network2, d1TestData, "d1 (test)");
        ClassificationMetrics testMetrics3 = evaluateModel(network3, d2TestData, "d2 (test)");
        
        double f1_d1 = testMetrics2.macroF1;
        double f1_d2 = testMetrics3.macroF1;
        double finalScore = calculateFinalScore(f1_d1, f1_d2);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "F1-Score (d1 test): " << f1_d1 << "\n";
        std::cout << "F1-Score (d2 test): " << f1_d2 << "\n";
        std::cout << "\nФормула: 0.5 * F1(d1) + 0.5 * F1(d2)\n";
        std::cout << "Итоговая оценка: " << finalScore << "\n";
        
        if (finalScore >= 0.55) {
            std::cout << "\n✓ ЗАЧЕТ! Итоговая оценка >= 0.55\n";
        } else {
            std::cout << "\n✗ НЕЗАЧЕТ. Итоговая оценка < 0.55\n";
            std::cout << "Требуется улучшение модели.\n";
        }
        
        std::cout << std::string(60, '=') << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nОШИБКА: " << e.what() << "\n";
        std::cerr << "\nВозможные причины:\n";
        std::cerr << "1. Файлы train.csv, d1.csv, d2.csv и test.csv не найдены в папке data/\n";
        std::cerr << "2. Некорректный формат CSV\n";
        std::cerr << "3. Недостаточно памяти\n";
        std::cerr << "4. Ошибка в конфигурации алгоритма\n";
        return 1;
    }
    return 0;
}