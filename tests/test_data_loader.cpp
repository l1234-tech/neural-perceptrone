#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>
#include "data/DataLoader.h"
#include "data/DataSet.h"

bool test_data_loader() {
    std::cout << "\nTesting DataLoader...\n";

    try {
        // 1. Создание временного CSV файла для тестирования
        std::string test_filename = "test_data_loader_temp.csv";
        {
            std::ofstream file(test_filename);
            file << "feature1,feature2,label\n";
            file << "0.1,0.2,0\n";
            file << "0.7,0.8,1\n";
            file << "0.3,0.4,0\n";
            file << "0.9,0.1,1\n";
        }
        std::cout << "1. Created test CSV file: " << test_filename << "\n";

        // 2. Тестирование статического метода loadFromCSV
        DataSet dataset = DataLoader::loadFromCSV(test_filename);
        std::cout << "2. loadFromCSV OK (size = " << dataset.size() << ")\n";
        assert(dataset.size() == 4);
        
        if (!dataset.empty()) {
            auto point = dataset.get(0);
            auto features = point.getFeatures();
            assert(features.size() == 2);
            std::cout << "   First point features: " << features[0] << ", " << features[1] 
                      << ", label: " << point.getLabel() << "\n";
        }

        // 3. Тестирование статического метода saveToCSV
        std::string save_filename = "test_data_loader_save.csv";
        DataLoader::saveToCSV(dataset, save_filename);
        std::cout << "3. saveToCSV OK (saved to " << save_filename << ")\n";
        
        // Проверяем, что файл создан
        assert(std::filesystem::exists(save_filename));

        // 4. Тестирование loadFromCSVNoLabels
        std::string test_nolabels_filename = "test_data_loader_nolabels_temp.csv";
        {
            std::ofstream file(test_nolabels_filename);
            file << "feature1,feature2\n";
            file << "0.1,0.2\n";
            file << "0.7,0.8\n";
        }
        
        DataSet dataset_nolabels = DataLoader::loadFromCSVNoLabels(test_nolabels_filename);
        std::cout << "4. loadFromCSVNoLabels OK (size = " << dataset_nolabels.size() << ")\n";
        assert(dataset_nolabels.size() == 2);

        // 5. Тестирование saveToCSVNoLabels
        std::string save_nolabels_filename = "test_data_loader_save_nolabels.csv";
        DataLoader::saveToCSVNoLabels(dataset_nolabels, save_nolabels_filename);
        std::cout << "5. saveToCSVNoLabels OK (saved to " << save_nolabels_filename << ")\n";
        assert(std::filesystem::exists(save_nolabels_filename));

        // 6. Тестирование detectFeatureCount
        size_t feature_count = DataLoader::detectFeatureCount(test_filename);
        std::cout << "6. detectFeatureCount OK (count = " << feature_count << ")\n";
        assert(feature_count == 2);

        // 7. Тестирование hasHeaderRow
        bool has_header = DataLoader::hasHeaderRow(test_filename);
        std::cout << "7. hasHeaderRow OK (has header = " << (has_header ? "true" : "false") << ")\n";
        assert(has_header == true);

        // 8. Тестирование конструктора DataLoader с параметрами
        DataLoader loader(test_filename, true, ',');
        DataSet loaded_dataset = loader.load();
        std::cout << "8. DataLoader constructor and load() OK (size = " << loaded_dataset.size() << ")\n";
        assert(loaded_dataset.size() == 4);

        // 9. Тестирование save и saveNoLabels
        DataLoader saver("test_data_loader_saver.csv", true, ',');
        saver.save(loaded_dataset);
        std::cout << "9. DataLoader.save() OK\n";
        
        DataLoader saver_nolabels("test_data_loader_saver_nolabels.csv", false, ',');
        saver_nolabels.saveNoLabels(loaded_dataset);
        std::cout << "   DataLoader.saveNoLabels() OK\n";

        // Очистка временных файлов
        std::filesystem::remove(test_filename);
        std::filesystem::remove(save_filename);
        std::filesystem::remove(test_nolabels_filename);
        std::filesystem::remove(save_nolabels_filename);
        std::filesystem::remove("test_data_loader_saver.csv");
        std::filesystem::remove("test_data_loader_saver_nolabels.csv");

        std::cout << "\nDataLoader tests passed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        
        // Попытка очистки временных файлов в случае ошибки
        try {
            std::filesystem::remove("test_data_loader_temp.csv");
            std::filesystem::remove("test_data_loader_save.csv");
            std::filesystem::remove("test_data_loader_nolabels_temp.csv");
            std::filesystem::remove("test_data_loader_save_nolabels.csv");
            std::filesystem::remove("test_data_loader_saver.csv");
            std::filesystem::remove("test_data_loader_saver_nolabels.csv");
        } catch (...) {}
        
        return false;
    }
}