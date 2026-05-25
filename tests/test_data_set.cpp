#include <iostream>
#include <cassert>
#include "data/DataSet.h"
#include "data/DataPoint.h"
#include "neural_network/NeuralNetwork.h"

bool test_data_set() {
    std::cout << "\nTesting DataSet...\n";

    try {
        // 1. Создание пустого DataSet
        DataSet empty_set;
        std::cout << "1. Empty DataSet created OK\n";
        assert(empty_set.empty());
        assert(empty_set.size() == 0);

        // 2. Добавление точек
        DataSet dataset;
        dataset.add(DataPoint({0.0, 0.0}, 0));
        dataset.add(DataPoint({1.0, 1.0}, 0));
        dataset.add(DataPoint({0.0, 1.0}, 1));
        dataset.add(DataPoint({1.0, 0.0}, 1));
        
        std::cout << "2. Added 4 points OK (size = " << dataset.size() << ")\n";
        assert(dataset.size() == 4);
        assert(!dataset.empty());

        // 3. Проверка доступа к точкам
        const DataPoint& point0 = dataset.get(0);
        auto features0 = point0.getFeatures();
        int label0 = point0.getLabel();
        
        std::cout << "3. Point access OK\n";
        std::cout << "   Point 0: features = [" << features0[0] << ", " << features0[1] 
                  << "], label = " << label0 << "\n";
        
        assert(features0.size() == 2);
        assert(features0[0] == 0.0 && features0[1] == 0.0);
        assert(label0 == 0);

        // 4. Проверка getFeatureCount
        size_t feature_count = dataset.getFeatureCount();
        std::cout << "4. getFeatureCount OK (count = " << feature_count << ")\n";
        assert(feature_count == 2);

        // 5. Проверка split (80/20)
        auto [train_set, test_set] = dataset.split(0.8);
        std::cout << "5. split(0.8) OK\n";
        std::cout << "   Train set size: " << train_set.size() << "\n";
        std::cout << "   Test set size: " << test_set.size() << "\n";
        
        assert(train_set.size() + test_set.size() == dataset.size());
        assert(train_set.size() == 3 || train_set.size() == 4); // 80% от 4 = 3.2 -> 3

        // 6. Проверка shuffle (проверяем только что не падает)
        DataSet shuffled = dataset;
        shuffled.shuffle();
        std::cout << "6. shuffle() OK (size after shuffle = " << shuffled.size() << ")\n";
        assert(shuffled.size() == dataset.size());

        // 7. Проверка computeAccuracy (нужна нейронная сеть)
        // Создаем простую сеть и проверяем, что метод вызывается без ошибок
        NeuralNetwork net({2, 2, 1});
        double accuracy = dataset.computeAccuracy(net);
        std::cout << "7. computeAccuracy OK (accuracy = " << accuracy << ")\n";
        assert(accuracy >= 0.0 && accuracy <= 1.0);

        // 8. Проверка split с разными соотношениями
        auto [set50, set50_2] = dataset.split(0.5);
        std::cout << "8. split(0.5) OK\n";
        std::cout << "   First half size: " << set50.size() << "\n";
        std::cout << "   Second half size: " << set50_2.size() << "\n";
        assert(set50.size() + set50_2.size() == dataset.size());

        // 9. Проверка split с ratio = 0.0 и 1.0
        auto [all_set, empty_set2] = dataset.split(1.0);
        std::cout << "9. split(1.0) OK\n";
        std::cout << "   All set size: " << all_set.size() << "\n";
        std::cout << "   Empty set size: " << empty_set2.size() << "\n";
        assert(all_set.size() == dataset.size());
        assert(empty_set2.empty());

        auto [empty_set3, all_set2] = dataset.split(0.0);
        std::cout << "   split(0.0) OK\n";
        std::cout << "   Empty set size: " << empty_set3.size() << "\n";
        std::cout << "   All set size: " << all_set2.size() << "\n";
        assert(empty_set3.empty());
        assert(all_set2.size() == dataset.size());

        // 10. Проверка копирования и перемещения
        DataSet copy_set = dataset;
        std::cout << "10. Copy constructor OK (size = " << copy_set.size() << ")\n";
        assert(copy_set.size() == dataset.size());

        DataSet moved_set = std::move(copy_set);
        std::cout << "    Move constructor OK (moved size = " << moved_set.size() 
                  << ", original size after move = " << copy_set.size() << ")\n";
        assert(moved_set.size() == dataset.size());

        std::cout << "\nDataSet tests passed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
}