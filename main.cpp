#include "neural_network/NeuralNetwork.h"
#include "trainer/GeneticTrainer.h"
#include "data/DataLoader.h"
#include "data/XORDataGenerator.h"
#include "utils/Random.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <filesystem>

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(50, '=') << "\n";
}

std::pair<DataSet, DataSet> loadOrGenerateData(bool useFiles = false) {
    printHeader("ПОДГОТОВКА ДАННЫХ");
    
    DataSet trainData, testData;
    
    // Создаем папку data если ее нет
    std::filesystem::create_directories("data");
    
    if (useFiles && std::filesystem::exists("data/train.csv") && 
                    std::filesystem::exists("data/test.csv")) {
        std::cout << "Загрузка данных из файлов...\n";
        
        try {
            trainData = DataLoader::loadFromCSV("data/train.csv");
            testData = DataLoader::loadFromCSV("data/test.csv");
            
            std::cout << "Обучающая выборка: " << trainData.size() << " примеров\n";
            std::cout << "Тестовая выборка: " << testData.size() << " примеров\n";
            
            return {trainData, testData};
            
        } catch (const std::exception& e) {
            std::cerr << "Ошибка загрузки данных: " << e.what() << "\n";
            std::cout << "Переключение на генерацию данных...\n";
            useFiles = false;
        }
    }
    
    // Генерация новых данных
    std::cout << "Генерация разнообразных данных XOR...\n";
    
    // УЛУЧШЕННЫЕ ПАРАМЕТРЫ: меньше шума, сбалансированные классы
    XORDataGenerator generator(
        300,    // размер обучающей выборки
        100,    // размер тестовой выборки
        0.05,   // уровень шума УМЕНЬШЕНО (было 0.15)
        0.5,    // соотношение классов СБАЛАНСИРОВАНО (было 0.6)
        42      // seed для воспроизводимости
    );
    
    auto [generatedTrain, generatedTest] = generator.generateData();
    trainData = generatedTrain;
    testData = generatedTest;
    
    generator.printStats(trainData, testData);
    
    std::cout << "\nСохранение данных в файлы...\n";
    // ИСПРАВЛЕННЫЙ ВЫЗОВ: передаем сгенерированные данные
    generator.saveToCSV(trainData, testData, "data/train.csv", "data/test.csv");
    
    return {trainData, testData};
}

// Функция обучения с улучшенными параметрами
NeuralNetwork trainModel(const DataSet& trainData, const DataSet& testData) {
    printHeader("ОБУЧЕНИЕ МОДЕЛИ");
    
    // УЛУЧШЕННАЯ АРХИТЕКТУРА: немного сложнее для лучшего обучения
    NeuralNetwork network({2, 8, 4, 1}); 
    
    std::cout << "Архитектура сети: [2 → 8 → 4 → 1]\n";
    std::cout << "Всего параметров: " << network.getTotalParameters() << "\n";
    
    // УЛУЧШЕННЫЕ ПАРАМЕТРЫ GeneticTrainer
    GeneticTrainer trainer(
        120,    // размер популяции (увеличено)
        120,    // максимальное число поколений
        0.15,   // вероятность мутации (оптимизировано)
        0.8,    // вероятность скрещивания
        0.15    // доля элитных особей (оптимизировано)
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

// Функция тестирования с расширенной статистикой
void testModel(const NeuralNetwork& network, const DataSet& testData) {
    printHeader("ТЕСТИРОВАНИЕ МОДЕЛИ");
    
    if (testData.empty()) {
        std::cout << "Тестовая выборка пуста\n";
        return;
    }
    
    int correct = 0;
    int falsePositives = 0;  // Класс 0 предсказан как 1
    int falseNegatives = 0;  // Класс 1 предсказан как 0
    int truePositives = 0;   // Класс 1 предсказан как 1
    int trueNegatives = 0;   // Класс 0 предсказан как 0
    
    std::vector<std::tuple<Vector, int, int, double>> errors; // вход, ожидаемый, предсказанный, уверенность
    
    for (size_t i = 0; i < testData.size(); ++i) {
        const auto& point = testData.get(i);
        const auto& features = point.getFeatures();
        
        Vector input(features.size());
        for (size_t j = 0; j < features.size(); ++j) {
            input[j] = features[j];
        }
        
        int predicted = network.predict(input);
        int expected = point.getLabel();
        Vector output = network.forward(input);
        double confidence = std::abs(output[0] - 0.5) * 2.0 * 100.0;
        
        if (predicted == expected) {
            correct++;
            if (expected == 0) trueNegatives++;
            else truePositives++;
        } else {
            if (expected == 0 && predicted == 1) falsePositives++;
            if (expected == 1 && predicted == 0) falseNegatives++;
            errors.emplace_back(input, expected, predicted, confidence);
        }
    }
    
    double accuracy = static_cast<double>(correct) / testData.size() * 100.0;
    double loss = network.computeLoss(testData);
    
    std::cout << "Точность: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    std::cout << "Ошибка (loss): " << std::setprecision(6) << loss << "\n";
    std::cout << "Правильных: " << correct << "/" << testData.size() << "\n";
    
    // Расширенная статистика
    if (testData.size() > 0) {
        double precision = (truePositives + falsePositives) > 0 ? 
            static_cast<double>(truePositives) / (truePositives + falsePositives) * 100.0 : 0.0;
        double recall = (truePositives + falseNegatives) > 0 ?
            static_cast<double>(truePositives) / (truePositives + falseNegatives) * 100.0 : 0.0;
        double f1Score = (precision + recall) > 0 ? 
            2.0 * precision * recall / (precision + recall) : 0.0;
        
        std::cout << "\nРасширенная статистика:\n";
        std::cout << "  Precision: " << std::setprecision(1) << precision << "%\n";
        std::cout << "  Recall: " << recall << "%\n";
        std::cout << "  F1-Score: " << f1Score << "%\n";
        std::cout << "  True Positives: " << truePositives << "\n";
        std::cout << "  True Negatives: " << trueNegatives << "\n";
        std::cout << "  False Positives: " << falsePositives << "\n";
        std::cout << "  False Negatives: " << falseNegatives << "\n";
    }
    
}

// Главная функция
int main() {
    try {
        // Инициализация
        initRandom(42);  // Фиксированный seed для воспроизводимости
        
        // Для Windows: установка кодировки UTF-8 (можно закомментировать если не Windows)
        #ifdef _WIN32
        system("chcp 65001 > nul");
        #endif

        printHeader("НЕЙРОННАЯ СЕТЬ ДЛЯ XOR");
        
        // 1. Подготовка данных (генерация или загрузка)
        // false - всегда генерировать новые данные
        // true - загружать из файлов если они существуют
        bool useExistingFiles = false;  
        auto [trainData, testData] = loadOrGenerateData(useExistingFiles);
        
        // 2. Обучение модели
        NeuralNetwork model = trainModel(trainData, testData);
        
        // 3. Тестирование модели
        testModel(model, testData);
        
    } catch (const std::exception& e) {
        std::cerr << "\nОШИБКА: " << e.what() << "\n";
        std::cerr << "\nВозможные причины:\n";
        std::cerr << "1. Проблемы с генерацией данных\n";
        std::cerr << "2. Недостаточно памяти\n";
        std::cerr << "3. Ошибка в конфигурации алгоритма\n";
        std::cerr << "4. Ошибка в вычислении функции потерь\n";
    }
    return 0;
}