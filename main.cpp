#include "neural_network/NeuralNetwork.h"
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

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

NeuralNetwork createNetwork(size_t featureCount) {
    // Автоматическое определение архитектуры: [featureCount, 8, 4, 1]
    std::vector<size_t> architecture = {featureCount, 8, 4, 1};
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

std::pair<DataSet, DataSet> loadAndSplitDataset(const std::string& filename) {
    std::cout << "\nЗагрузка данных из: " << filename << "\n";
    
    DataSet fullDataset = DataLoader::loadFromCSV(filename);
    size_t featureCount = fullDataset.getFeatureCount();
    
    std::cout << "Загружено примеров: " << fullDataset.size() << "\n";
    std::cout << "Количество признаков: " << featureCount << "\n";
    
    // Разделение 80/20
    auto [trainData, testData] = fullDataset.split(0.8);
    
    std::cout << "Обучающая выборка: " << trainData.size() << " примеров\n";
    std::cout << "Тестовая выборка: " << testData.size() << " примеров\n";
    
    return {trainData, testData};
}

NeuralNetwork trainModel(NeuralNetwork& network, const DataSet& trainData, const DataSet& testData) {
    printHeader("ОБУЧЕНИЕ МОДЕЛИ");
    
    GeneticTrainer trainer(
        300,   // размер популяции 
        200,   // максимальное число поколений 
        0.15,  // вероятность мутации 
        0.8,    // вероятность скрещивания
        0.15    // доля элитных особей 
    );
    
    std::cout << "\nКонфигурация алгоритма:\n" << trainer.getConfig() << "\n";
    
    std::cout << "\nНачало обучения...\n";
    auto startTime = std::chrono::steady_clock::now();
    
    auto lossHistory = trainer.train(network, trainData, testData);
    
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
                std::cout << "  Поколение " << std::setw(3) << i 
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
        
        // Датасет 1 (2 признака)
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ДАТАСЕТ 1 (d1.csv - 2 признака)\n";
        std::cout << std::string(60, '#') << "\n";
        
        auto [trainData1, testData1] = loadAndSplitDataset("data/d1.csv");
        NeuralNetwork network1 = createNetwork(trainData1.getFeatureCount());
        trainModel(network1, trainData1, testData1);
        ClassificationMetrics metrics1 = evaluateModel(network1, testData1, "d1.csv");
        
        // Датасет 2 (4 признака)
        std::cout << "\n" << std::string(60, '#') << "\n";
        std::cout << "# ДАТАСЕТ 2 (d2.csv - 4 признака)\n";
        std::cout << std::string(60, '#') << "\n";
        
        auto [trainData2, testData2] = loadAndSplitDataset("data/d2.csv");
        NeuralNetwork network2 = createNetwork(trainData2.getFeatureCount());
        trainModel(network2, trainData2, testData2);
        ClassificationMetrics metrics2 = evaluateModel(network2, testData2, "d2.csv");
        
        // Итоговая оценка
        printHeader("ИТОГОВАЯ ОЦЕНКА");
        
        double f1_d1 = metrics1.macroF1;
        double f1_d2 = metrics2.macroF1;
        double finalScore = calculateFinalScore(f1_d1, f1_d2);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "F1-Score (d1): " << f1_d1 << "\n";
        std::cout << "F1-Score (d2): " << f1_d2 << "\n";
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
        std::cerr << "1. Файлы d1.csv и d2.csv не найдены в папке data/\n";
        std::cerr << "2. Некорректный формат CSV\n";
        std::cerr << "3. Недостаточно памяти\n";
        std::cerr << "4. Ошибка в конфигурации алгоритма\n";
        return 1;
    }
    return 0;
}
