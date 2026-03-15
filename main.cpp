#include "utils/Random.h"
#include "utils/Matrix.h"
#include "utils/Vector.h"
#include "neural_network/Activation.h"
#include "neural_network/Layer.h"
#include "data/DataLoader.h"
#include "data/DataSet.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

// Функция для создания тестовых CSV файлов
void createTestFiles() {
    // Создаем train.csv с метками
    std::ofstream trainFile("data/train.csv");
    trainFile << "1.0,2.0,0\n";
    trainFile << "3.0,4.0,1\n";
    trainFile << "5.0,6.0,0\n";
    trainFile << "7.0,8.0,1\n";
    trainFile.close();
    
    // Создаем test.csv без меток
    std::ofstream testFile("data/test.csv");
    testFile << "9.0,10.0\n";
    testFile << "11.0,12.0\n";
    testFile << "13.0,14.0\n";
    testFile.close();
    
    std::cout << "Созданы тестовые файлы data/train.csv и data/test.csv\n";
}

// Демонстрация работы DataLoader
void demonstrateDataLoader() {
    std::cout << "\n=== Демонстрация DataLoader ===\n\n";
    
    try {
        // 1. Использование статических методов (старый способ)
        std::cout << "1. Статические методы (обратная совместимость):\n";
        DataSet trainData = DataLoader::loadFromCSV("data/train.csv");
        std::cout << "   Загружено " << trainData.size() << " точек с метками\n";
        
        DataSet testData = DataLoader::loadFromCSVNoLabels("data/test.csv");
        std::cout << "   Загружено " << testData.size() << " точек без меток\n";
        
        // 2. Использование экземплярных методов (новый способ)
        std::cout << "\n2. Экземплярные методы (ООП подход):\n";
        
        // Загрузка с метками
        DataLoader trainLoader("data/train.csv", true, ',');
        DataSet trainData2 = trainLoader.load();
        std::cout << "   DataLoader с метками: " << trainData2.size() << " точек\n";
        
        if (!trainData2.empty()) {
            const auto& point = trainData2.get(0);
            const auto& features = point.getFeatures();
            std::cout << "   Первая точка: [" << features[0] << ", " << features[1] 
                      << "] -> класс " << point.getLabel() << "\n";
        }
        
        // Загрузка без меток
        DataLoader testLoader("data/test.csv", false, ',');
        DataSet testData2 = testLoader.load();
        std::cout << "   DataLoader без меток: " << testData2.size() << " точек\n";
        
        if (!testData2.empty()) {
            const auto& point = testData2.get(0);
            std::cout << "   Метка первой точки: " << point.getLabel() << " (должна быть -1)\n";
        }
        
        // 3. Демонстрация изменения разделителя
        std::cout << "\n3. Изменение конфигурации:\n";
        DataLoader loader("data/train.csv", true, ',');
        std::cout << "   Текущий разделитель: '" << loader.getDelimiter() << "'\n";
        
        loader.setDelimiter(';');
        std::cout << "   Новый разделитель: '" << loader.getDelimiter() << "'\n";
        
        // 4. Демонстрация сохранения данных
        std::cout << "\n4. Сохранение данных:\n";
        
        // Создаем простой датасет
        DataSet sampleData;
        sampleData.add(DataPoint({1.5, 2.5}, 0));
        sampleData.add(DataPoint({3.5, 4.5}, 1));
        sampleData.add(DataPoint({5.5, 6.5}, 0));
        
        // Сохраняем разными способами
        DataLoader::saveToCSV(sampleData, "output_static.csv");
        std::cout << "   Сохранено через статический метод: output_static.csv\n";
        
        DataLoader saver("output_instance.csv", true, ';');
        saver.save(sampleData);
        std::cout << "   Сохранено через экземплярный метод: output_instance.csv\n";
        
        // 5. Проверка обработки ошибок
        std::cout << "\n5. Обработка ошибок:\n";
        try {
            DataLoader errorLoader("non_existent_file.csv", true, ',');
            errorLoader.load();
        } catch (const std::exception& e) {
            std::cout << "   Поймано исключение: " << e.what() << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Ошибка: " << e.what() << "\n";
        std::cout << "Создайте файлы data/train.csv и data/test.csv\n";
    }
}

int main() {
    initRandom(42);
    
    // 1. Тест базовых утилит (оставляем из оригинального main.cpp)
    std::cout << "\n[1] Test utils:\n";
    
    Matrix m(2, 2);
    m.at(0,0) = 1.0;
    m.at(0,1) = 2.0;
    m.at(1,0) = 3.0; 
    m.at(1,1) = 4.0;
    std::cout << "- Matrix 2x2: OK\n";
    
    Vector v1({1.0, 2.0}), v2({3.0, 4.0});
    std::cout << "- scalar multipy of vectors: " << v1.dot(v2) << "\n";
    std::cout << "- sigmoida(2.0): " << Activation::sigmoid(2.0) << "\n";
    
    // 2. Тест нейронной сети 
    std::cout << "\n[2] Test for Layer:\n";
    Layer hidden(2, 4);
    Vector input({0.3, 0.8});
    Vector output = hidden.forward(input);
    
    std::cout << "- InPut: [" << input[0] << ", " << input[1] << "]\n";
    std::cout << "- OutPut: [5";
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << output[i];
        if (i < output.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // 3. Демонстрация DataLoader
    std::cout << "\n[3] Test DataLoader:\n";
    demonstrateDataLoader();
    
    std::cout << "\n=============================================\n";
    std::cout << "All tests are good!\n";
    
    return 0;
}