#include "neural_network/NeuralNetwork.h"
#include "trainer/GradientTrainer.h"
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

// Функция для создания обучающих и тестовых файлов (80/20)
void createTrainTestFiles() {
    std::cout << "\nСоздание обучающих и тестовых файлов (80/20)...\n";
    
    try {
        // Обработка d1.csv
        std::cout << "\nОбработка d1.csv:\n";
        DataSet d1FullDataset = DataLoader::loadFromCSV("data/d1.csv");
        auto [d1TrainData, d1TestData] = d1FullDataset.split(0.8);
        
        // Сохраняем обучающую часть с метками
        DataLoader::saveToCSV(d1TrainData, "data/d1_train.csv");
        std::cout << "Создан файл: data/d1_train.csv (80% с метками, " << d1TrainData.size() << " примеров)\n";
        
        // Сохраняем тестовую часть без меток (для предсказаний)
        DataLoader::saveToCSVNoLabels(d1TestData, "data/d1_test.csv");
        std::cout << "Создан файл: data/d1_test.csv (20% без меток, " << d1TestData.size() << " примеров)\n";
        
        // Сохраняем тестовую часть с метками (для оценки)
        DataLoader::saveToCSV(d1TestData, "data/d1_test_labels.csv");
        std::cout << "Создан файл: data/d1_test_labels.csv (20% с метками, " << d1TestData.size() << " примеров)\n";
        
        // Обработка d2.csv
        std::cout << "\nОбработка d2.csv:\n";
        DataSet d2FullDataset = DataLoader::loadFromCSV("data/d2.csv");
        auto [d2TrainData, d2TestData] = d2FullDataset.split(0.8);
        
        // Сохраняем обучающую часть с метками
        DataLoader::saveToCSV(d2TrainData, "data/d2_train.csv");
        std::cout << "Создан файл: data/d2_train.csv (80% с метками, " << d2TrainData.size() << " примеров)\n";
        
        // Сохраняем тестовую часть без меток (для предсказаний)
        DataLoader::saveToCSVNoLabels(d2TestData, "data/d2_test.csv");
        std::cout << "Создан файл: data/d2_test.csv (20% без меток, " << d2TestData.size() << " примеров)\n";
        
        // Сохраняем тестовую часть с метками (для оценки)
        DataLoader::saveToCSV(d2TestData, "data/d2_test_labels.csv");
        std::cout << "Создан файл: data/d2_test_labels.csv (20% с метками, " << d2TestData.size() << " примеров)\n";
        
        // Обработка train.csv (опционально)
        std::cout << "\nОбработка train.csv:\n";
        DataSet trainFullDataset = DataLoader::loadFromCSV("data/train.csv");
        auto [trainTrainData, trainTestData] = trainFullDataset.split(0.8);
        
        // Сохраняем обучающую часть с метками
        DataLoader::saveToCSV(trainTrainData, "data/train_train.csv");
        std::cout << "Создан файл: data/train_train.csv (80% с метками, " << trainTrainData.size() << " примеров)\n";
        
        // Сохраняем тестовую часть без меток
        DataLoader::saveToCSVNoLabels(trainTestData, "data/train_test.csv");
        std::cout << "Создан файл: data/train_test.csv (20% без меток, " << trainTestData.size() << " примеров)\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при создании файлов: " << e.what() << "\n";
        throw; // Пробрасываем дальше
    }
}

NeuralNetwork trainModel(NeuralNetwork& network, const DataSet& trainData, const std::string& datasetName = "") {
    printHeader("ОБУЧЕНИЕ МОДЕЛИ" + (datasetName.empty() ? "" : " (" + datasetName + ")"));
    
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
        std::cout << "\nИспользуются консервативные параметры для train.csv\n";
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
        std::cout << "\nИспользуются стандартные параметры для датасетов d1/d2\n";
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
        
        // Создаем обучающие и тестовые файлы (80/20)
        createTrainTestFiles();
        
        // Датасет train.csv (2 признака) - используем обучающую часть
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ДАТАСЕТ train.csv (2 признака) - ОБУЧЕНИЕ\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Загружаем обу��ающие данные (80%)
        DataSet trainTrainData = loadDataset("data/train_train.csv", true);
        NeuralNetwork network1 = createNetwork(trainTrainData.getFeatureCount(), "train");
        trainModel(network1, trainTrainData, "train.csv");
        
        // Датасет d1.csv (2 признака) - используем обучающую часть
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ДАТАСЕТ d1.csv (2 признака) - ОБУЧЕНИЕ\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Загружаем обучающие данные (80%)
        DataSet d1TrainData = loadDataset("data/d1_train.csv", true);
        NeuralNetwork network2 = createNetwork(d1TrainData.getFeatureCount(), "d1");
        trainModel(network2, d1TrainData, "d1.csv");
        
        // Датасет d2.csv (4 признака) - используем обучающую часть
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ДАТАСЕТ d2.csv (4 признака) - ОБУЧЕНИЕ\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Загружаем обучающие данные (80%)
        DataSet d2TrainData = loadDataset("data/d2_train.csv", true);
        NeuralNetwork network3 = createNetwork(d2TrainData.getFeatureCount(), "d2");
        trainModel(network3, d2TrainData, "d2.csv");
        
        // ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ФАЙЛАХ БЕЗ МЕТОК (20%)
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ФАЙЛАХ БЕЗ МЕТОК (20%)\n";
        std::cout << std::string(60, '#') << "\n";
        
        // Предсказания на test.csv (2 признака, без меток)
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ test.csv (2 признака, предсказания)\n";
        std::cout << std::string(60, '-') << "\n";
        
        DataSet testDataset = loadDataset("data/test.csv", false); // без меток
        
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
        
        // Предсказания на d1_test.csv (20% без меток)
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ d1_test.csv (20% без меток, 2 признака)\n";
        std::cout << std::string(60, '-') << "\n";
        
        try {
            DataSet d1TestDataset = loadDataset("data/d1_test.csv", false);
            if (network2.getArchitecture()[0] == d1TestDataset.getFeatureCount()) {
                std::cout << "\nПредсказания для d1_test.csv (первые 10 примеров):\n";
                std::cout << "feature_0	feature_1	Предсказание (0/1)\n";
                for (size_t i = 0; i < std::min(size_t(10), d1TestDataset.size()); ++i) {
                    const auto& point = d1TestDataset.get(i);
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
                std::cout << "Всего предсказаний для d1_test.csv: " << d1TestDataset.size() << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Не удалось сделать предсказания для d1_test.csv: " << e.what() << "\n";
        }
        
        // Предсказания на d2_test.csv (20% без меток)
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "ДАТАСЕТ d2_test.csv (20% без меток, 4 признака)\n";
        std::cout << std::string(60, '-') << "\n";
        
        try {
            DataSet d2TestDataset = loadDataset("data/d2_test.csv", false);
            if (network3.getArchitecture()[0] == d2TestDataset.getFeatureCount()) {
                std::cout << "\nПредсказания для d2_test.csv (первые 10 примеров):\n";
                std::cout << "feature_0	feature_1	feature_2	feature_3	Предсказание (0/1)\n";
                for (size_t i = 0; i < std::min(size_t(10), d2TestDataset.size()); ++i) {
                    const auto& point = d2TestDataset.get(i);
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
                std::cout << "Всего предсказаний для d2_test.csv: " << d2TestDataset.size() << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Не удалось сделать предсказания для d2_test.csv: " << e.what() << "\n";
        }
        
        // Итоговая оценка (на основе тестовых данных d1 и d2 с метками)
        printHeader("ИТОГОВАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ (20%)");
        
        // Загружаем тестовые данные с метками (для оценки)
        DataSet d1TestDataWithLabels = loadDataset("data/d1_test_labels.csv", true);
        DataSet d2TestDataWithLabels = loadDataset("data/d2_test_labels.csv", true);
        
        ClassificationMetrics testMetrics2 = evaluateModel(network2, d1TestDataWithLabels, "d1 (test)");
        ClassificationMetrics testMetrics3 = evaluateModel(network3, d2TestDataWithLabels, "d2 (test)");
        
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