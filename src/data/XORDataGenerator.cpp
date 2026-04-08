#include "data/XORDataGenerator.h"
#include "data/DataLoader.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <filesystem>

XORDataGenerator::XORDataGenerator(size_t trainSize, 
                                  size_t testSize,
                                  double noiseLevel,
                                  double classRatio,
                                  int seed)
    : trainSize(trainSize),
      testSize(testSize),
      noiseLevel(noiseLevel),
      classRatio(classRatio),
      seed(seed) {
    
    if (trainSize < 50) {
        throw std::invalid_argument("Размер обучающей выборки должен быть не менее 50");
    }
    if (testSize < 20) {
        throw std::invalid_argument("Размер тестовой выборки должен быть не менее 20");
    }
}

std::pair<DataSet, DataSet> XORDataGenerator::generateData() {
    // Устанавливаем seed для воспроизводимости
    initRandom(seed);
    
    DataSet trainData = generateTrainData();
    DataSet testData = generateTestData();
    
    return {trainData, testData};
}

DataSet XORDataGenerator::generateTrainData() {
    return generateDataset(trainSize, true);
}

DataSet XORDataGenerator::generateTestData() {
    return generateDataset(testSize, false);
}

DataSet XORDataGenerator::generateDataset(size_t size, bool isTrain) {
    DataSet dataset;
    
    // Определяем количество точек каждого класса
    size_t class0Count = static_cast<size_t>(size * classRatio);
    size_t class1Count = size - class0Count;
    
    // Генерируем точки класса 0 (оба низкие или оба высокие)
    for (size_t i = 0; i < class0Count; ++i) {
        // Случайно выбираем, генерировать ли точку в нижнем левом или верхнем правом квадранте
        bool lowerLeft = randomDouble(0.0, 1.0) < 0.5;
        
        auto [x, y] = generatePoint(0); // Класс 0
        int label = calculateLabel(x, y, isTrain && randomDouble(0.0, 1.0) < noiseLevel);
        
        dataset.add(DataPoint({x, y}, label));
    }
    
    // Генерируем точки класса 1 (разные)
    for (size_t i = 0; i < class1Count; ++i) {
        auto [x, y] = generatePoint(1); // Класс 1
        int label = calculateLabel(x, y, isTrain && randomDouble(0.0, 1.0) < noiseLevel);
        
        dataset.add(DataPoint({x, y}, label));
    }
    
    // Добавляем специальные точки (только в обучающую выборку)
    if (isTrain) {
        addSpecialPoints(dataset, size / 10); // 10% специальных точек
    }
    
    // Перемешиваем данные
    // Note: DataSet не имеет метода shuffle, поэтому создадим временный вектор
    std::vector<DataPoint> points;
    for (size_t i = 0; i < dataset.size(); ++i) {
        points.push_back(dataset.get(i));
    }
    
    // Простое перемешивание
    for (size_t i = 0; i < points.size(); ++i) {
        size_t j = randomInt(0, points.size() - 1);
        if (i != j) {
            std::swap(points[i], points[j]);
        }
    }
    
    // Создаем новый DataSet с перемешанными точки
    DataSet shuffledDataset;
    for (const auto& point : points) {
        shuffledDataset.add(point);
    }
    
    return shuffledDataset;
}

std::pair<double, double> XORDataGenerator::generatePoint(bool targetClass) {
    double x, y;
    
    if (targetClass == 0) { // Класс 0: оба низкие или оба высокие
        if (randomDouble(0.0, 1.0) < 0.5) {
            // Нижний левый квадрант (оба < 0.5)
            x = randomDouble(0.0, 0.45);
            y = randomDouble(0.0, 0.45);
        } else {
            // Верхний правый квадрант (оба >= 0.5)
            x = randomDouble(0.55, 1.0);
            y = randomDouble(0.55, 1.0);
        }
    } else { // Класс 1: разные
        if (randomDouble(0.0, 1.0) < 0.5) {
            // x низкий, y высокий
            x = randomDouble(0.0, 0.45);
            y = randomDouble(0.55, 1.0);
        } else {
            // x высокий, y низкий
            x = randomDouble(0.55, 1.0);
            y = randomDouble(0.0, 0.45);
        }
    }
    
    // Добавляем небольшой случайный сдвиг для разнообразия
    x += randomDouble(-0.05, 0.05);
    y += randomDouble(-0.05, 0.05);
    
    // Ограничиваем значения [0, 1]
    x = std::max(0.0, std::min(1.0, x));
    y = std::max(0.0, std::min(1.0, y));
    
    return {x, y};
}

int XORDataGenerator::calculateLabel(double x, double y, bool addNoise) {
    // Чистая XOR логика
    bool isClass0 = (x < 0.5 && y < 0.5) || (x >= 0.5 && y >= 0.5);
    
    // Добавляем шум если нужно
    if (addNoise) {
        // С вероятностью noiseLevel меняем класс
        if (randomDouble(0.0, 1.0) < noiseLevel) {
            isClass0 = !isClass0;
        }
    }
    
    return isClass0 ? 0 : 1;
}

void XORDataGenerator::addSpecialPoints(DataSet& dataset, size_t count) {
    // Добавляем специальные точки для лучшего обучения
    
    // 1. Точки точно на границе 0.5
    for (size_t i = 0; i < count / 4; ++i) {
        double offset = randomDouble(-0.1, 0.1);
        double x = 0.5 + offset;
        double y = 0.5 + (randomDouble(0.0, 1.0) < 0.5 ? offset : -offset);
        
        x = std::max(0.0, std::min(1.0, x));
        y = std::max(0.0, std::min(1.0, y));
        
        int label = calculateLabel(x, y, false);
        dataset.add(DataPoint({x, y}, label));
    }
    
    // 2. Точки на осях
    for (size_t i = 0; i < count / 4; ++i) {
        double val = randomDouble(0.0, 1.0);
        if (randomDouble(0.0, 1.0) < 0.5) {
            // x = 0 или 1
            double x = randomDouble(0.0, 1.0) < 0.5 ? 0.0 : 1.0;
            double y = val;
            int label = calculateLabel(x, y, false);
            dataset.add(DataPoint({x, y}, label));
        } else {
            // y = 0 или 1
            double x = val;
            double y = randomDouble(0.0, 1.0) < 0.5 ? 0.0 : 1.0;
            int label = calculateLabel(x, y, false);
            dataset.add(DataPoint({x, y}, label));
        }
    }
    
    // 3. Точки в центре каждого квадранта
    std::vector<std::pair<double, double>> centers = {
        {0.25, 0.25}, // нижний левый
        {0.25, 0.75}, // верхний левый  
        {0.75, 0.25}, // нижний правый
        {0.75, 0.75}  // верхний правый
    };
    
    for (const auto& [cx, cy] : centers) {
        for (size_t i = 0; i < count / 16; ++i) {
            double x = cx + randomDouble(-0.2, 0.2);
            double y = cy + randomDouble(-0.2, 0.2);
            
            x = std::max(0.0, std::min(1.0, x));
            y = std::max(0.0, std::min(1.0, y));
            
            int label = calculateLabel(x, y, false);
            dataset.add(DataPoint({x, y}, label));
        }
    }
}

void XORDataGenerator::saveToCSV(const DataSet& trainData, 
                                const DataSet& testData,
                                const std::string& trainPath,
                                const std::string& testPath) {
    // Создаем папку data, если она не существует
    std::filesystem::path trainFilePath(trainPath);
    std::filesystem::path testFilePath(testPath);
    
    std::filesystem::create_directories(trainFilePath.parent_path());
    std::filesystem::create_directories(testFilePath.parent_path());
    
    // Сохраняем обучающие данные
    std::ofstream trainFile(trainPath);
    if (!trainFile.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + trainPath);
    }
    
    trainFile << "x1,x2,label\n";
    for (size_t i = 0; i < trainData.size(); ++i) {
        const auto& point = trainData.get(i);
        const auto& features = point.getFeatures();
        trainFile << std::fixed << std::setprecision(4) 
                 << features[0] << "," << features[1] << "," << point.getLabel() << "\n";
    }
    trainFile.close();
    
    // Сохраняем тестовые данные
    std::ofstream testFile(testPath);
    if (!testFile.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + testPath);
    }
    
    testFile << "x1,x2,label\n";
    for (size_t i = 0; i < testData.size(); ++i) {
        const auto& point = testData.get(i);
        const auto& features = point.getFeatures();
        testFile << std::fixed << std::setprecision(4) 
                << features[0] << "," << features[1] << "," << point.getLabel() << "\n";
    }
    testFile.close();
    
    std::cout << "Данные сохранены в:\n";
    std::cout << "  " << trainPath << " (" << trainData.size() << " точек)\n";
    std::cout << "  " << testPath << " (" << testData.size() << " точек)\n";
}

void XORDataGenerator::printStats(const DataSet& trainData, const DataSet& testData) const {
    std::cout << "\nСтатистика данных:\n";
    std::cout << "Обучающая выборка: " << trainData.size() << " точек\n";
    std::cout << "Тестовая выборка: " << testData.size() << " точек\n";
    
    // Статистика по классам для обучающей выборки
    int trainClass0 = 0, trainClass1 = 0;
    for (size_t i = 0; i < trainData.size(); ++i) {
        if (trainData.get(i).getLabel() == 0) trainClass0++;
        else trainClass1++;
    }
    
    double trainRatio0 = static_cast<double>(trainClass0) / trainData.size() * 100.0;
    double trainRatio1 = static_cast<double>(trainClass1) / trainData.size() * 100.0;
    
    std::cout << "Баланс классов (обучение):\n";
    std::cout << "  Класс 0: " << trainClass0 << " точек (" 
              << std::fixed << std::setprecision(1) << trainRatio0 << "%)\n";
    std::cout << "  Класс 1: " << trainClass1 << " точек (" << trainRatio1 << "%)\n";
    
    // Статистика по классам для тестовой выборки
    int testClass0 = 0, testClass1 = 0;
    for (size_t i = 0; i < testData.size(); ++i) {
        if (testData.get(i).getLabel() == 0) testClass0++;
        else testClass1++;
    }
    
    double testRatio0 = static_cast<double>(testClass0) / testData.size() * 100.0;
    double testRatio1 = static_cast<double>(testClass1) / testData.size() * 100.0;
    
    std::cout << "Баланс классов (тест):\n";
    std::cout << "  Класс 0: " << testClass0 << " точек (" << testRatio0 << "%)\n";
    std::cout << "  Класс 1: " << testClass1 << " точек (" << testRatio1 << "%)\n";
    
    // Распределение по квадрантам (только для обучающей выборки)
    int q1 = 0, q2 = 0, q3 = 0, q4 = 0; // квадранты: 1-верхний правый, 2-верхний левый, 3-нижний левый, 4-нижний правый
    
    for (size_t i = 0; i < std::min<size_t>(100, trainData.size()); ++i) {
        const auto& point = trainData.get(i);
        const auto& features = point.getFeatures();
        double x = features[0];
        double y = features[1];
        
        if (x >= 0.5 && y >= 0.5) q1++;
        else if (x < 0.5 && y >= 0.5) q2++;
        else if (x < 0.5 && y < 0.5) q3++;
        else q4++;
    }
    
    std::cout << "Распределение по квадрантам (первые 100 точек):\n";
    std::cout << "  Верхний правый: " << q1 << " точек\n";
    std::cout << "  Верхний левый: " << q2 << " точек\n";
    std::cout << "  Нижний левый: " << q3 << " точек\n";
    std::cout << "  Нижний правый: " << q4 << " точек\n";
}